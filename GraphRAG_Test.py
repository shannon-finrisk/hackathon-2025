import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import re
from typing import Any

# Load sample dataset
news = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv")[:20]  # Reduced for testing

# Convert data into LlamaIndex Document objects
documents = [
    Document(text=f"{row['title']}: {row['text']}")
    for _, row in news.iterrows()
]

splitter = SentenceSplitter(
    chunk_size=512,  # Reduced chunk size for better extraction
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="llama3.2:3b",  # Using a smaller model for this Hackathon but we can scale it up if necessary. 
    request_timeout=6000.0,
)

# Simplified patterns that are easier for the LLM to follow
entity_pattern = r'ENTITY:\s*(.+?)\s*TYPE:\s*(.+?)\s*DESC:\s*(.+?)(?=\s*ENTITY:|\s*RELATIONSHIP:|\s*$)'
relationship_pattern = r'RELATIONSHIP:\s*(.+?)\s*->\s*(.+?)\s*TYPE:\s*(.+?)\s*DESC:\s*(.+?)(?=\s*ENTITY:|\s*RELATIONSHIP:|\s*$)'

def parse_fn(response_str: str) -> Any:
    """More robust parsing function"""
    print(f"Raw LLM response: {response_str}")  # Debug: see what the LLM is producing
    
    entities = re.findall(entity_pattern, response_str, re.DOTALL)
    relationships = re.findall(relationship_pattern, response_str, re.DOTALL)
    
    print(f"Extracted entities: {entities}")  # Debug
    print(f"Extracted relationships: {relationships}")  # Debug
    
    return entities, relationships

import asyncio
import nest_asyncio
nest_asyncio.apply()

from typing import Any, List, Callable, Optional, Union, Dict
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TransformComponent, BaseNode

class GraphRAGExtractor(TransformComponent):
    """Fixed GraphRAG extractor with better error handling"""

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_fn,
        max_paths_per_chunk: int = 5,  # Increased for better coverage
        num_workers: int = 2,  # Reduced for stability
    ) -> None:
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node with better error handling"""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        print(f"Processing text: {text[:100]}...")  # Debug
        
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            print(f"LLM Response: {llm_response}")  # Debug
            
            entities, entities_relationship = self.parse_fn(llm_response)
            print(f"Parsed - Entities: {len(entities)}, Relationships: {len(entities_relationship)}")  # Debug
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        
        # Add entities
        for entity_name, entity_type, description in entities:
            entity_node = EntityNode(
                name=entity_name.strip(),
                label=entity_type.strip(),
                properties={"description": description.strip()}
            )
            existing_nodes.append(entity_node)

        # Add relationships
        for source, target, relation, description in entities_relationship:
            source_node = EntityNode(name=source.strip(), properties={})
            target_node = EntityNode(name=target.strip(), properties={})
            
            rel_node = Relation(
                label=relation.strip(),
                source_id=source_node.id,
                target_id=target_node.id,
                properties={"description": description.strip()},
            )

            existing_nodes.extend([source_node, target_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        
        print(f"Final - Nodes: {len(existing_nodes)}, Relations: {len(existing_relations)}")  # Debug
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

# Simplified prompt that's easier for the LLM to follow
KG_TRIPLET_EXTRACT_TMPL = """
Please analyze the following text and extract entities and relationships.

TEXT: {text}

INSTRUCTIONS:
1. Extract important ENTITIES. For each entity, provide:
ENTITY: [entity name]
TYPE: [person, organization, location, event, concept, object]
DESC: [brief description]

2. Extract RELATIONSHIPS between entities. For each relationship, provide:
RELATIONSHIP: [source entity] -> [target entity]
TYPE: [relationship type like works_for, located_in, part_of, involved_in, etc.]
DESC: [brief description of the relationship]

IMPORTANT: Use the exact format above. Extract up to {max_knowledge_triplets} of the most important entities and relationships.

EXAMPLES:
ENTITY: Barack Obama
TYPE: person
DESC: Former President of the United States

RELATIONSHIP: Barack Obama -> United States
TYPE: president_of
DESC: Served as the 44th President

Now analyze this text:
{text}
"""

kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=3,
    parse_fn=parse_fn,
    num_workers=1,  # Start with 1 worker for debugging
)

from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.llms import ChatMessage

class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = Ollama(model="llama3.2:3b", request_timeout=6000.0).chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response
    
    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.ollama import OllamaEmbedding

embeddings = OllamaEmbedding(
    model_name="llama3.2:3b",  # Match the LLM model
)

# Create index with debug info
print("Creating PropertyGraphIndex...")
index = PropertyGraphIndex(
    nodes=nodes[:10],  # Start with just 10 nodes for testing
    llm=llm,
    embed_model=embeddings,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)

PERSIST_DIR = "Hackathon"
from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
)

# Store the index so that it can be loaded in later if necessary. Uncomment the code below to load in the index. 
index.storage_context.persist(persist_dir=PERSIST_DIR)
# graph_store = GraphRAGStore.from_persist_dir(PERSIST_DIR)

# storage_context = StorageContext.from_defaults(
#     persist_dir=PERSIST_DIR,
#     property_graph_store=graph_store
# )

# index = load_index_from_storage(storage_context, llm=llm, embed_model=embeddings)

# Check if we have any data in the graph
print(f"Graph store nodes: {len(index.property_graph_store.graph.nodes)}")
print(f"Graph store relations: {len(index.property_graph_store.graph.relations)}")

# Only build communities if we have data
if (index.property_graph_store.graph.nodes and 
    index.property_graph_store.graph.relations):
    print("Building communities...")
    index.property_graph_store.build_communities()
    
    # Print community summaries
    summaries = index.property_graph_store.get_community_summaries()
    print("Community Summaries:")
    for community_id, summary in summaries.items():
        print(f"Community {community_id}: {summary}")
else:
    print("No graph data extracted. Check the LLM extraction output.")


from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from IPython.display import Markdown, display
class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"QUERY: {query}\n\n"
            f"COMMUNITY SUMMARY: {community_summary}\n\n"
            "INSTRUCTIONS:\n"
            "1. If the community summary contains information that answers the query, provide a concise answer.\n"
            "2. If the community summary does NOT contain relevant information to answer the query, output exactly: null\n"
            "3. Do not make up information or try to be helpful beyond what's in the summary.\n\n"
            "OUTPUT FORMAT: Either a direct answer or 'null'"
        )
        
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="Provide your response based on the instructions above."),
        ]
        
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        
        # Force null if the response indicates no relevance
        
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

llm = Ollama(
    model="gemma3:4b",  # Using a more reliable model for extraction
    request_timeout=6000.0,
)
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, llm=llm
)
# with open('graphrag_query_engine.pkl', 'rb') as f:
#         import pickle
#         query_engine = pickle.load(f)

response = query_engine.query("What is the latest news in the football transfer market.")
print(f"{response.response}")
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import re
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from google import genai

# Load markdown file
with open("src2/novel.md", "r", encoding="utf-8") as f:
    novel_text = f.read()

# Create a single document from the entire text
documents = [Document(text=novel_text)]

# Use SentenceSplitter to chunk the document
splitter = SentenceSplitter(
    chunk_size=1024,  # Reduced chunk size for better extraction
    chunk_overlap=20,
)

nodes = splitter.get_nodes_from_documents(documents)

from llama_index.llms.ollama import Ollama
from llama_index.llms.vertex import Vertex
from llama_index.core.llms import ChatMessage, MessageRole
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

GCP_PROJECT = "finrisk-sandbox"
GCP_LOCATION = "us-central1"
vertexai.init(project= GCP_PROJECT, location="us-central1")
# Configure safety settings to be more permissive
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = Vertex(model="gemini-2.5-flash-lite", safety_settings=safety_settings, max_tokens=10000)


# llm = Ollama(
#     model="qwen3:4b-instruct",  # Using a smaller model for this Hackathon but we can scale it up if necessary. 
#     request_timeout=600000.0,
# )

# client = genai.Client(
#     vertexai=True, project='finrisk-sandbox', location='us-central1'
# )

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
-Goal-
Given a text document, identify all important entities and their types, and extract all relationships among the identified entities.
Extract up to {max_knowledge_triplets} of the most important entity-relation triplets.

-Steps-
1. Identify all important ENTITIES from the text. For each entity, extract:
   - Entity name: Name of the entity (capitalized, use full names when possible)
   - Entity type: One of: person, organization, location, event, concept, object
   - Entity description: Comprehensive description of the entity's attributes, role, and activities

   Format each entity as:
   ENTITY: [entity name]
   TYPE: [entity type]
   DESC: [entity description]

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly and directly related* to each other.
   Only extract relationships where there is an explicit connection mentioned in the text.

   For each pair of related entities, extract:
   - Source entity: name of the source entity (exactly as identified in step 1)
   - Target entity: name of the target entity (exactly as identified in step 1)
   - Relationship type: Specific relationship type (e.g., works_for, located_in, part_of, involved_in, created_by, member_of, related_to, etc.)
   - Relationship description: Clear explanation of why these entities are related, based on the text

   Format each relationship as:
   RELATIONSHIP: [source entity] -> [target entity]
   TYPE: [relationship type]
   DESC: [relationship description]

3. Prioritize the most important and clearly stated entities and relationships. Do not extract vague or inferred connections.

-Examples-

Example 1:
ENTITY: Francis Bacon
TYPE: person
DESC: English philosopher and statesman, born in 1560

RELATIONSHIP: Francis Bacon -> Trinity College
TYPE: attended
DESC: Studied at Trinity College Cambridge

Example 2:
ENTITY: Trinity College
TYPE: organization
DESC: University college in Cambridge where Francis Bacon studied

RELATIONSHIP: Trinity College -> Cambridge
TYPE: located_in
DESC: Trinity College is located in the city of Cambridge

-Important Notes-
- Use the EXACT format shown above with ENTITY:, TYPE:, DESC:, RELATIONSHIP:, and -> symbols
- Entity names must match exactly between entities and relationships
- Only extract relationships that are explicitly stated or clearly implied in the text
- Focus on quality over quantity - extract the most important connections
- If an entity appears multiple times, use the same name consistently

-Real Data-
TEXT: {text}

Now analyze this text and extract entities and relationships following the format above:
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
    
    def get_community_graph(self):
        """Creates a simplified graph where nodes are communities."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        
        # Create mapping: node -> community_id
        community_mapping = {item.node: item.cluster for item in community_hierarchical_clusters}
        
        # Create simplified community graph
        community_graph = nx.Graph()
        
        # Add community nodes with summaries
        community_nodes = {}
        for cluster_id in set(community_mapping.values()):
            summary = self.community_summary.get(cluster_id, f"Community {cluster_id}")
            # Truncate summary for display
            short_summary = summary[:100] + "..." if len(summary) > 100 else summary
            community_nodes[cluster_id] = short_summary
            community_graph.add_node(cluster_id, summary=summary, label=f"Community {cluster_id}")
        
        # Add edges between communities (if nodes in different communities are connected)
        inter_community_edges = {}
        for u, v, data in nx_graph.edges(data=True):
            comm_u = community_mapping.get(u)
            comm_v = community_mapping.get(v)
            
            if comm_u != comm_v:  # Edge between different communities
                if (comm_u, comm_v) not in inter_community_edges:
                    inter_community_edges[(comm_u, comm_v)] = []
                inter_community_edges[(comm_u, comm_v)].append(data.get('relationship', 'unknown'))
        
        # Add edges to community graph (weight = number of connections)
        for (comm_u, comm_v), relationships in inter_community_edges.items():
            community_graph.add_edge(comm_u, comm_v, 
                                    weight=len(relationships),
                                    relationships=relationships)
        
        return community_graph, community_nodes
    
    def get_communities_at_level(self, level=0):
        """
        Extract communities at a specific hierarchical level.
        Level 0 = root/coarse communities (larger)
        Level 1 = sub-communities (finer)
        Higher levels = even finer granularity
        """
        nx_graph = self._create_nx_graph()
        
        # For level 0, use a larger max_cluster_size to get coarser communities
        # For level 1, use smaller max_cluster_size to get finer communities
        if level == 0:
            # Root level: larger communities (less granular)
            max_size = max(50, len(nx_graph.nodes()) // 5)  # Adaptive based on graph size
        elif level == 1:
            # Level 1: finer communities (more granular)
            max_size = self.max_cluster_size  # Use your existing setting
        else:
            # Even finer levels
            max_size = max(2, self.max_cluster_size // (level + 1))
        
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=max_size
        )
        
        # Create mapping: node -> community_id at this level
        community_mapping = {item.node: item.cluster for item in community_hierarchical_clusters}
        
        return community_mapping, nx_graph
    
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
        with open("community_summaries.pkl", "rb") as f:
          import pickle
          self.community_summary = pickle.load(f)
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
    nodes=nodes[:5],  # Start with just 10 nodes for testing
    llm=llm,
    embed_model=embeddings,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)

PERSIST_DIR = "Novel"
from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
)

# Store the index so that it can be loaded in later if necessary. Uncomment the code below to load in the index. 

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
    index.storage_context.persist(persist_dir=PERSIST_DIR)
   
else:
    print("No graph data extracted. Check the LLM extraction output.")

#index.storage_context.persist(persist_dir=PERSIST_DIR)

# Visualize the simplified community graph

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
    model="qwen3:4b-instruct",  # Using a more reliable model for extraction
    request_timeout=6000.0,
)
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, llm=llm
)
# with open('graphrag_query_engine.pkl', 'rb') as f:
#         import pickle
#         query_engine = pickle.load(f)

response = query_engine.query("Where did Francis Bacon study?")
print(f"{response.response}")

# Get community graph
community_graph, community_nodes = index.property_graph_store.get_community_graph()

if community_graph.number_of_nodes() > 0:
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(community_graph, k=2, iterations=50)
    
    # Get node sizes based on number of connections (degree)
    node_sizes = [community_graph.degree(node) * 1000 + 2000 for node in community_graph.nodes()]
    
    # Color nodes by community ID (using a colormap)
    node_colors = [hash(str(node)) % 256 for node in community_graph.nodes()]
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(node_colors))]
    
    # Draw nodes
    nx.draw_networkx_nodes(community_graph, pos, 
                          node_color=colors,
                          node_size=node_sizes,
                          alpha=0.8,
                          ax=ax)
    
    # Draw edges with thickness based on weight (number of connections)
    edges = community_graph.edges()
    edge_weights = [community_graph[u][v].get('weight', 1) for u, v in edges]
    nx.draw_networkx_edges(community_graph, pos,
                          width=[w * 0.5 for w in edge_weights],
                          alpha=0.6,
                          edge_color='gray',
                          ax=ax)
    
    # Create labels with community summaries
    labels = {}
    for node in community_graph.nodes():
        summary = community_nodes.get(node, f"Community {node}")
        # Create multi-line label (first line: ID, second: summary)
        labels[node] = f"C{node}\n{summary[:50]}..."
    
    nx.draw_networkx_labels(community_graph, pos, labels, 
                           font_size=8, font_weight='bold', ax=ax)
    
    # Add edge labels showing relationship types
    edge_labels = {}
    for u, v, data in community_graph.edges(data=True):
        relationships = data.get('relationships', [])
        unique_rels = list(set(relationships))[:3]  # Show up to 3 unique relationship types
        edge_labels[(u, v)] = f"{len(relationships)} links\n{', '.join(unique_rels)}"
    
    nx.draw_networkx_edge_labels(community_graph, pos, edge_labels, 
                                font_size=6, ax=ax)
    
    plt.title("Simplified Community Graph\n(Nodes = Communities, Edges = Inter-community connections)", 
              size=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("COMMUNITY GRAPH STATISTICS")
    print("="*60)
    print(f"Number of communities: {community_graph.number_of_nodes()}")
    print(f"Inter-community connections: {community_graph.number_of_edges()}")
    print(f"\nCommunity Details:")
    for node in sorted(community_graph.nodes()):
        degree = community_graph.degree(node)
        summary = community_nodes.get(node, "No summary")
        print(f"\n  Community {node}:")
        print(f"    Connections to other communities: {degree}")
        print(f"    Summary: {summary[:10]}...")
else:
    print("No communities found in the graph.")
def visualize_hierarchical_communities(index, level_0_max_size=None, level_1_max_size=None):
    """
    Create side-by-side visualization of communities at two hierarchical levels.
    Similar to the research paper figure.
    """
    nx_graph = index.property_graph_store._create_nx_graph()
    
    # Get communities at level 0 (coarse/root)
    if level_0_max_size is None:
        level_0_max_size = max(50, len(nx_graph.nodes()) // 5)
    clusters_level_0 = hierarchical_leiden(nx_graph, max_cluster_size=level_0_max_size)
    mapping_level_0 = {item.node: item.cluster for item in clusters_level_0}
    
    # Get communities at level 1 (finer)
    if level_1_max_size is None:
        level_1_max_size = index.property_graph_store.max_cluster_size
    clusters_level_1 = hierarchical_leiden(nx_graph, max_cluster_size=level_1_max_size)
    mapping_level_1 = {item.node: item.cluster for item in clusters_level_1}
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Use the same layout for both graphs for comparison
    pos = nx.spring_layout(nx_graph, k=1, iterations=50, seed=42)
    
    # Get unique communities for coloring
    communities_0 = set(mapping_level_0.values())
    communities_1 = set(mapping_level_1.values())
    
    # Create color maps
    cmap_0 = cm.get_cmap('tab20')
    cmap_1 = cm.get_cmap('tab20')
    colors_0 = {comm: cmap_0(i % 20) for i, comm in enumerate(communities_0)}
    colors_1 = {comm: cmap_1(i % 20) for i, comm in enumerate(communities_1)}
    
    # Add fallback color for unmapped nodes
    default_color = (0.5, 0.5, 0.5, 1.0)  # Gray
    colors_0[-1] = default_color
    colors_1[-1] = default_color
    
    # Plot Level 0 (Root communities) - Graph (a)
    node_colors_0 = []
    for node in nx_graph.nodes():
        comm_id = mapping_level_0.get(node, -1)
        node_colors_0.append(colors_0.get(comm_id, default_color))
    
    nx.draw_networkx_nodes(nx_graph, pos, 
                          node_color=node_colors_0,
                          node_size=50,
                          alpha=0.8,
                          ax=ax1,
                          edgecolors='black',
                          linewidths=0.5)
    nx.draw_networkx_edges(nx_graph, pos,
                          alpha=0.2,
                          edge_color='gray',
                          width=0.5,
                          ax=ax1)
    ax1.set_title(f'(a) Root communities at level 0\n({len(communities_0)} communities, max_size={level_0_max_size})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Plot Level 1 (Sub-communities) - Graph (b)
    node_colors_1 = []
    for node in nx_graph.nodes():
        comm_id = mapping_level_1.get(node, -1)
        node_colors_1.append(colors_1.get(comm_id, default_color))
    
    nx.draw_networkx_nodes(nx_graph, pos, 
                          node_color=node_colors_1,
                          node_size=50,
                          alpha=0.8,
                          ax=ax2,
                          edgecolors='black',
                          linewidths=0.5)
    nx.draw_networkx_edges(nx_graph, pos,
                          alpha=0.2,
                          edge_color='gray',
                          width=0.5,
                          ax=ax2)
    ax2.set_title(f'(b) Sub-communities at level 1\n({len(communities_1)} communities, max_size={level_1_max_size})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    plt.suptitle('Hierarchical Community Detection', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("HIERARCHICAL COMMUNITY STATISTICS")
    print("="*60)
    print(f"\nLevel 0 (Root):")
    print(f"  Number of communities: {len(communities_0)}")
    print(f"  Average community size: {len(nx_graph.nodes()) / len(communities_0):.1f} nodes")
    
    print(f"\nLevel 1 (Sub-communities):")
    print(f"  Number of communities: {len(communities_1)}")
    print(f"  Average community size: {len(nx_graph.nodes()) / len(communities_1):.1f} nodes")
    
    # Show how level 0 communities are subdivided
    print(f"\nCommunity Subdivision Analysis:")
    comm_0_to_1 = {}
    for node in nx_graph.nodes():
        comm_0 = mapping_level_0.get(node)
        comm_1 = mapping_level_1.get(node)
        if comm_0 not in comm_0_to_1:
            comm_0_to_1[comm_0] = set()
        comm_0_to_1[comm_0].add(comm_1)
    
    # for comm_0, sub_comms in sorted(comm_0_to_1.items())[:5]:  # Show first 5
    #     print(f"  Level 0 Community {comm_0} contains {len(sub_comms)} sub-communities at Level 1")
# Call the visualization function
visualize_hierarchical_communities(index)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

def visualize_original_graph(index):
    """
    Visualize the original knowledge graph with all entities (nodes) and relationships (edges).
    """
    # Get the NetworkX graph from the index
    nx_graph = index.property_graph_store._create_nx_graph()
    
    if nx_graph.number_of_nodes() == 0:
        print("Graph is empty - no nodes to visualize.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Use a layout algorithm (spring layout works well for general graphs)
    # Adjust k parameter for spacing (larger = more spread out)
    pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=42)
    
    # Calculate node sizes based on degree (more connections = larger node)
    degrees = dict(nx_graph.degree())
    node_sizes = [degrees[node] * 300 + 200 for node in nx_graph.nodes()]
    
    # Color nodes by degree (more connected = darker)
    node_colors = [degrees[node] for node in nx_graph.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        nx_graph, 
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis,  # Color map: darker = more connections
        alpha=0.8,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        alpha=0.3,
        edge_color='gray',
        width=1.0,
        ax=ax
    )
    
    # Draw node labels (entity names)
    # Only show labels for nodes with high degree to avoid clutter
    high_degree_nodes = {node: label for node, label in zip(nx_graph.nodes(), nx_graph.nodes()) 
                         if degrees[node] >= 3}  # Only label nodes with 3+ connections
    
    if high_degree_nodes:
        nx.draw_networkx_labels(
            nx_graph,
            pos,
            labels=high_degree_nodes,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
    
    # Add colorbar for node degree
    plt.colorbar(nodes, ax=ax, label='Number of Connections (Degree)')
    
    plt.title(f'Original Knowledge Graph\n({nx_graph.number_of_nodes()} entities, {nx_graph.number_of_edges()} relationships)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Total entities (nodes): {nx_graph.number_of_nodes()}")
    print(f"Total relationships (edges): {nx_graph.number_of_edges()}")
    print(f"Average connections per entity: {sum(degrees.values()) / len(degrees):.2f}")
    print(f"\nMost connected entities:")
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    for i, (node, degree) in enumerate(sorted_degrees[:10], 1):
        print(f"  {i}. {node}: {degree} connections")
    
    return nx_graph

# Call the function
visualize_original_graph(index)

def visualize_original_graph_detailed(index, max_nodes=100):
    """
    Visualize with relationship type labels and entity details.
    If graph is too large, only show a subset.
    """
    nx_graph = index.property_graph_store._create_nx_graph()
    
    if nx_graph.number_of_nodes() == 0:
        print("Graph is empty.")
        return
    
    # If graph is too large, create a subgraph with most connected nodes
    if nx_graph.number_of_nodes() > max_nodes:
        print(f"Graph has {nx_graph.number_of_nodes()} nodes. Showing top {max_nodes} most connected nodes.")
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_list = [node for node, _ in top_nodes]
        nx_graph = nx_graph.subgraph(top_node_list)
    
    # Create mapping from node IDs to clean names
    node_id_to_name = {}
    for node_obj in index.property_graph_store.graph.nodes.values():
        node_id = str(node_obj.id) if hasattr(node_obj, 'id') else str(node_obj)
        node_name = node_obj.name if hasattr(node_obj, 'name') else str(node_obj)
        node_id_to_name[node_id] = node_name
    
    # Create a new graph with clean node names
    clean_graph = nx.Graph()
    for node_id in nx_graph.nodes():
        clean_name = node_id_to_name.get(node_id, node_id)
        clean_graph.add_node(clean_name)
    
    # Add edges with clean labels
    for u, v, data in nx_graph.edges(data=True):
        u_clean = node_id_to_name.get(u, u)
        v_clean = node_id_to_name.get(v, v)
        rel_label = data.get('relationship', 'related_to')
        clean_graph.add_edge(u_clean, v_clean, relationship=rel_label)
    
    fig, ax = plt.subplots(figsize=(24, 18))
    
    pos = nx.spring_layout(clean_graph, k=2, iterations=50, seed=42)
    
    # Node sizes and colors
    degrees = dict(clean_graph.degree())
    node_sizes = [degrees[node] * 400 + 300 for node in clean_graph.nodes()]
    node_colors = [degrees[node] for node in clean_graph.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        clean_graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.plasma,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        clean_graph, pos,
        alpha=0.4,
        edge_color='gray',
        width=1.5,
        ax=ax
    )
    
    # Add edge labels (relationship types) - only for a subset to avoid clutter
    edge_labels = {}
    for u, v, data in list(clean_graph.edges(data=True))[:50]:  # Show first 50 edges
        rel = data.get('relationship', '')
        if rel:
            edge_labels[(u, v)] = rel[:15]  # Truncate long names
    
    nx.draw_networkx_edge_labels(
        clean_graph, pos,
        edge_labels=edge_labels,
        font_size=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        ax=ax
    )
    
    # Add node labels (entity names)
    nx.draw_networkx_labels(
        clean_graph, pos,
        font_size=7,
        font_weight='bold',
        ax=ax
    )
    
    plt.colorbar(nodes, ax=ax, label='Number of Connections')
    plt.title(f'Original Knowledge Graph (Detailed View)\n{clean_graph.number_of_nodes()} entities, {clean_graph.number_of_edges()} relationships',
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Call it
visualize_original_graph_detailed(index)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_community_wordclouds(index):
    """Create word clouds for each community summary."""
    summaries = index.property_graph_store.get_community_summaries()
    
    n_communities = len(summaries)
    cols = 3
    rows = (n_communities + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    axes = axes.flatten() if n_communities > 1 else [axes]
    
    for idx, (comm_id, summary) in enumerate(summaries.items()):
        if idx >= len(axes):
            break
            
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             max_words=50).generate(summary)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'Community {comm_id}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(summaries), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

create_community_wordclouds(index)

# Install: pip install wordcloud

from neo4j import GraphDatabase
import numpy as np
from typing import List
from ..models import KnowledgeGraph

class GraphIntegrator:
    """
    A class to integrate and manage graph data in a Neo4j database.
    """
    def __init__(self, uri: str, username: str, password: str):
        """
        Initializes the GraphIntegrator with database connection parameters.
        
        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for database access.
            password (str): Password for database access.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = self.connect()
        
    def connect(self):
        """
        Establishes a connection to the Neo4j database.
        
        Returns:
            A Neo4j driver instance for executing queries.
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return driver

    def run_query(self, query: str):
        """
        Runs a Cypher query against the Neo4j database.
        
        Args:
            query (str): The Cypher query to run.
        """
        session = self.driver.session()
        try:
            session.run(query)
        finally:
            session.close()
            
    @staticmethod
    def transform_embeddings_to_str_list(embeddings: np.array) -> str:
        """
        Transforms a NumPy array of embeddings into a comma-separated string.
        
        Args:
            embeddings (np.array): An array of embeddings.
        
        Returns:
            str: A comma-separated string of embeddings.
        """
        if embeddings is None:
            return ""
        return ",".join(list(embeddings.astype("str")))
    
    @staticmethod
    def transform_str_list_to_embeddings(embeddings: List[str]) -> np.array:
        """
        Transforms a comma-separated string of embeddings back into a NumPy array.
        
        Args:
            embeddings (str): A comma-separated string of embeddings.
        
        Returns:
            np.array: A NumPy array of embeddings.
        """
        if embeddings is None:
            return ""
        return np.array(embeddings.split(",")).astype(np.float64)
    
    @staticmethod
    def escape_str(s: str) -> str:
        """
        Escapes double quotes in a string for safe insertion into a Cypher query.
        """
        return s.replace('"', '\\"')
    
    @staticmethod
    def format_value(value) -> str:
        """
        Converts a value to a string and escapes it for safe Cypher insertion.
        """
        return GraphIntegrator.escape_str(str(value))
    
    def create_nodes(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating nodes in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing entities.
        
        Returns:
            List[str]: A list of Cypher queries for node creation.
        """
        queries = []
        for node in knowledge_graph.entities:
            # Escape the node name and label if needed.
            node_name = GraphIntegrator.format_value(node.name)
            node_label = node.label  # Assuming label is already valid
            
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value_str = GraphIntegrator.transform_embeddings_to_str_list(value)
                else:
                    value_str = GraphIntegrator.format_value(value)
                # Build a SET clause for each property.
                properties.append(f'SET n.{prop.replace(" ", "_")} = "{value_str}"')

            query = f'CREATE (n:{node_label} {{name: "{node_name}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    def create_relationships(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating relationships in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing relationships.
        
        Returns:
            List[str]: A list of Cypher queries for relationship creation.
        """
        rels = []
        for rel in knowledge_graph.relationships:
            # Escape start and end node names.
            start_label = rel.startEntity.label
            start_name = GraphIntegrator.format_value(rel.startEntity.name)
            end_label = rel.endEntity.label
            end_name = GraphIntegrator.format_value(rel.endEntity.name)
            rel_name = rel.name  # Assuming relationship type is valid
            
            property_statements = ' '.join(
                [
                    f'SET r.{key.replace(" ", "_")} = "{GraphIntegrator.transform_embeddings_to_str_list(value) if key=="embeddings" else GraphIntegrator.format_value(value)}"'
                    for key, value in rel.properties.model_dump().items()
                ]
            )
            
            query = (
                f'MATCH (n:{start_label} {{name: "{start_name}"}}), '
                f'(m:{end_label} {{name: "{end_name}"}}) '
                f'MERGE (n)-[r:{rel_name}]->(m) {property_statements}'
            )
            rels.append(query)
            
        return rels

    def visualize_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        """
        Runs the necessary queries to visualize a graph structure from a KnowledgeGraph input.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
        """
        nodes = self.create_nodes(knowledge_graph=knowledge_graph)
        relationships = self.create_relationships(knowledge_graph=knowledge_graph)
        
        for node_query in nodes:
            self.run_query(node_query)

        for rel_query in relationships:
            self.run_query(rel_query)
import concurrent.futures
from atom.models import KnowledgeGraph, Entity, Relationship
from atom.utils import Matcher, LangchainOutputParser, RelationshipsExtractor
import asyncio

from typing import List 

class Atom:
    def __init__(self, llm_model, embeddings_model) -> None:        
        """
        Initializes the iText2KG with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities and relationships from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations of extracted entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """

        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)

    def merge_two_kgs(self, kg1, kg2):
        """
        Merges two KGs using the same logic as the sequential approach above.
        Returns a single KnowledgeGraph.
        """
        updated_entities, updated_relationships = self.matcher.match_entities_and_update_relationships(
            entities_2=kg1.entities,
            relationships_2=kg1.relationships,
            entities_1=kg2.entities,
            relationships_1=kg2.relationships
        )
        return KnowledgeGraph(entities=updated_entities, relationships=updated_relationships)

    def parallel_atomic_merge(self, kgs, max_workers=4):
        """
        Merges a list of KnowledgeGraphs in parallel, reducing them pairwise.
        """
        # Keep merging until we have just one KG
        current = kgs
        while len(current) > 1:
            merged_results = []
            
            # Prepare pairs
            pairs = [(current[i], current[i+1]) 
                    for i in range(0, len(current) - 1, 2)]
            
            # If there's an odd KG out, keep it aside to append later
            leftover = current[-1] if len(current) % 2 == 1 else None
            
            # Merge pairs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.merge_two_kgs, p[0], p[1]) for p in pairs]
                for f in concurrent.futures.as_completed(futures):
                    merged_results.append(f.result())
            
            # Rebuild current list from newly merged KGs + leftover
            if leftover:
                merged_results.append(leftover)
            
            current = merged_results

        # At the end, current[0] is our big merged KG
        return current[0]
    
    async def build_atomic_kg_from_triplets(self, relationships:RelationshipsExtractor):
        temp_kg = KnowledgeGraph(entities=[Entity(**rel.startNode.model_dump()) for rel in relationships] + [Entity(**rel.endNode.model_dump()) for rel in relationships])
        await temp_kg.embed_entities(embeddings_function=self.langchain_output_parser.calculate_embeddings)

        embedded_relationships = [Relationship(name=relationship.name, 
                                    startEntity=temp_kg.get_entity(Entity(**relationship.startNode.model_dump())), 
                                    endEntity= temp_kg.get_entity(Entity(**relationship.endNode.model_dump())))
                        for relationship in relationships]
        

        kg = KnowledgeGraph(entities=temp_kg.entities, relationships=embedded_relationships)
        await kg.embed_relationships(embeddings_function=self.langchain_output_parser.calculate_embeddings)
        
        return kg

    async def build_graph(self, 
                          atomic_facts:List[str],
                          timestamp: str,
                          existing_knowledge_graph:KnowledgeGraph=None,
                          ent_threshold:float = 0.7,
                          rel_threshold:float = 0.7,
                          entity_name_weight:float=0.6,
                          entity_label_weight:float=0.4) -> KnowledgeGraph:
        print("[INFO] ------- Extracting Triplets")
        relationships = await self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure=RelationshipsExtractor, contexts=atomic_facts)
        print("[INFO] ------- Building Atomic KGs")
        atomic_kgs = await asyncio.gather(*list(map(self.build_atomic_kg_from_triplets, [relation.relationships for relation in relationships])))

        print("[INFO] ------- Adding Source Context to Atomic KGs")
        for atomic_kg, fact in zip(atomic_kgs, atomic_facts):
            atomic_kg.add_sources_to_relationships(source=fact)

        print("[INFO] ------- Merging Atomic KGs")
        cleaned_atomic_kgs = [kg for kg in atomic_kgs if kg.relationships != []]
        merged_kg = self.parallel_atomic_merge(cleaned_atomic_kgs)

        print("[INFO] ------- Adding Timestamps to Relationships")
        merged_kg.add_timestamps_to_relationships(timestamps=[timestamp])

        
        return merged_kg


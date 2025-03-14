from atom.models import KnowledgeGraph, Entity, Relationship, RelationshipProperties
from atom.utils import Matcher, LangchainOutputParser, RelationshipsExtractor
import asyncio
import dateparser
import concurrent.futures

from typing import List 
import time

class Atom:
    def __init__(self, llm_model, embeddings_model) -> None:        
        """
        Initializes the ATOM with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities and relationships from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations of extracted entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """

        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)

    def merge_two_kgs(self, kg1, kg2, rel_threshold:float=0.8, ent_threshold:float=0.8):
        """
        Merges two KGs using the same logic as the sequential approach above.
        Returns a single KnowledgeGraph.
        """
        updated_entities, updated_relationships = self.matcher.match_entities_and_update_relationships(
            entities_2=kg1.entities,
            relationships_2=kg1.relationships,
            entities_1=kg2.entities,
            relationships_1=kg2.relationships,
            rel_threshold=rel_threshold,
            ent_threshold=ent_threshold
        )
        return KnowledgeGraph(entities=updated_entities, relationships=updated_relationships)

    def parallel_atomic_merge(self, kgs, rel_threshold:float=0.8, ent_threshold:float=0.8, max_workers=4):
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
                futures = [executor.submit(self.merge_two_kgs, p[0], p[1], rel_threshold, ent_threshold) for p in pairs]
                for f in concurrent.futures.as_completed(futures):
                    merged_results.append(f.result())
            
            # Rebuild current list from newly merged KGs + leftover
            if leftover:
                merged_results.append(leftover)
            
            current = merged_results

        # At the end, current[0] is our big merged KG
        return current[0]
    
    async def build_atomic_kg_from_triplets(self, relationships:RelationshipsExtractor):
        embedded_relationships = []
        temp_kg = KnowledgeGraph(entities=[Entity(**rel.startNode.model_dump()) for rel in relationships] + [Entity(**rel.endNode.model_dump()) for rel in relationships])
        await temp_kg.embed_entities(embeddings_function=self.langchain_output_parser.calculate_embeddings)

        for relationship in relationships:
            if relationship.t_valid is None:
                relationship.t_valid = []
            elif relationship.t_invalid is None:
                relationship.t_invalid = []
            
            embedded_relationships.append(Relationship(name=relationship.name, 
                                        startEntity=temp_kg.get_entity(Entity(**relationship.startNode.model_dump())), 
                                        endEntity= temp_kg.get_entity(Entity(**relationship.endNode.model_dump())),
                                        properties = RelationshipProperties(t_valid=[dateparser.parse(ts).timestamp() for ts in relationship.t_valid], 
                                                                            t_invalid=[dateparser.parse(ts).timestamp() for ts in relationship.t_invalid])))
            
        

        kg = KnowledgeGraph(entities=temp_kg.entities, relationships=embedded_relationships)
        await kg.embed_relationships(embeddings_function=self.langchain_output_parser.calculate_embeddings)
        
        return kg

    async def build_graph(self, 
                          atomic_facts:List[str],
                          obs_timestamp: str,
                          existing_knowledge_graph:KnowledgeGraph=None,
                          ent_threshold:float = 0.8,
                          rel_threshold:float = 0.8,
                          entity_name_weight:float=0.7,
                          entity_label_weight:float=0.3,
                          max_workers:int=4,
                        ) -> KnowledgeGraph:
        system_query = f""" 
        Observation Time : {obs_timestamp}
        You are a top-tier algorithm designed for extracting information in structured 
        formats to build a knowledge graph.
        Try to capture as much information from the text as possible without 
        sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text
        Remember, the knowledge graph should be coherent and easily understandable, 
        so maintaining consistency in entity references is crucial.
        """
        examples = """ 
        FEW SHOT EXAMPLES \n
        * Michel served as CFO at Acme Corp from 2019 to 2021. He was hired by Beta Inc in 2021, but left that role in 2023. -> (michel, is_cfo, acme corp, [2019], [2021]), (michel, was_hired_by, beta inc, 2021, 2023)
        \n
        * Subsequent experiments confirmed the role of microRNAs in modulating cell growth. -> (experiments, confirmed_role, micrornas, [], []), (micrornas, modulate, cell growth, [], [])
        \n
        * Researchers used high-resolution imaging in a study on neural plasticity. -> (researchers, used, high-resolution imaging, [], []), (high-resolution imaging, used_in, study on neural plasticity, [], [])
        \n
        * Sarah was a board member of GreenFuture until 2019. -> (Sarah, is_board_member, greenfuture, [], [2019])
        \n
        * Dr. Lee was the head of the Oncology Department until 2022. -> (dr. lee, is_head_of, oncology department, [], [2022])
        \n
        * Activity-dependent modulation of receptor trafficking is crucial for maintaining synaptic efficacy. -> (activity-dependent modulation, involves, receptor trafficking, [], []), (receptor trafficking, maintains, synaptic efficacy, [], [])
        \n
        * (observation time = <observation_date>) John Doe is no longer the CEO of GreenIT a few months ago. -> (john doe, is_CEO, greenit, [], [<new observation_date by deducting 3months >])
        """
        print("[INFO] ------- Extracting Triplets")
        relationships = await self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure=RelationshipsExtractor, contexts=atomic_facts, system_query=system_query+examples)
        
        print("[INFO] ------- Building Atomic KGs")
        
        atomic_kgs = await asyncio.gather(*list(map(self.build_atomic_kg_from_triplets, [relation.relationships for relation in relationships])))
        print("[INFO] ------- Adding Source Context to Atomic KGs")
        for atomic_kg, fact in zip(atomic_kgs, atomic_facts):
            atomic_kg.add_sources_to_relationships(source=fact)

        print("[INFO] ------- Merging Atomic KGs")
        cleaned_atomic_kgs = [kg for kg in atomic_kgs if kg.relationships != []]
        merged_kg = self.parallel_atomic_merge(kgs=cleaned_atomic_kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)

        print("[INFO] ------- Adding Timestamps to Relationships")
        merged_kg.add_timestamps_to_relationships(timestamps=[obs_timestamp])
    
        if existing_knowledge_graph:
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities_1=merged_kg.entities,
                                                                 entities_2=existing_knowledge_graph.entities,
                                                                 relationships_1=merged_kg.relationships,
                                                                 relationships_2=existing_knowledge_graph.relationships,
                                                                 ent_threshold=ent_threshold,
                                                                 rel_threshold=rel_threshold,
                                                                 )    
        
            constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
            return constructed_kg
        return merged_kg
    
    async def build_graph_from_different_obs_times(self,
                                                   atomic_facts_with_obs_timestamps:dict,
                                                    existing_knowledge_graph:KnowledgeGraph=None,
                                                    ent_threshold:float = 0.8,
                                                    rel_threshold:float = 0.8,
                                                    entity_name_weight:float=0.8,
                                                    entity_label_weight:float=0.2,
                                                    max_workers:int=4,
                                               ):
        kgs = await asyncio.gather(*[
                        self.build_graph(
                            atomic_facts=atomic_facts_with_obs_timestamps[timestamp], 
                            obs_timestamp=timestamp,
                            ent_threshold=ent_threshold,
                            rel_threshold=rel_threshold,
                            entity_name_weight=entity_name_weight,
                            entity_label_weight=entity_label_weight,
                            existing_knowledge_graph=None,
                        ) for timestamp in atomic_facts_with_obs_timestamps
                    ])
        if existing_knowledge_graph:
            return self.parallel_atomic_merge(kgs=[existing_knowledge_graph] + kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)
        
        return self.parallel_atomic_merge(kgs=kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)
    
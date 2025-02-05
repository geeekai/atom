import re
import numpy as np
from typing import Callable, List, Union
from pydantic import BaseModel, Field, ConfigDict
import dateparser

# -------------------------------------------
# Create a common base model class
# -------------------------------------------
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="ignore"
    )

LABEL_PATTERN = re.compile(r'[^a-zA-Z0-9]+')  # For cleaning labels
NAME_PATTERN  = re.compile(r'[_"\-]+')       # For cleaning name underscores, quotes, dashes

# -------------------------------------------
# Entity/Relationship properties
# -------------------------------------------
class EntityProperties(BaseModelWithConfig):
    embeddings: np.ndarray = None

class RelationshipProperties(BaseModelWithConfig):
    embeddings: np.ndarray = None
    source:      str    = ""
    timestamps:   List[float]    = []

# -------------------------------------------
# Entity model
# -------------------------------------------
class Entity(BaseModelWithConfig):
    label: str = ""
    name: str  = ""
    properties: EntityProperties = Field(default_factory=EntityProperties)

    def process(self) -> "Entity":
        """
        Normalize `label` and `name` in-place and return self.
        """
        self.label = LABEL_PATTERN.sub("_", self.label).replace("&", "and")
        n = self.name.lower()
        n = NAME_PATTERN.sub(" ", n)
        self.name = n.strip()
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, Entity):
            return self.name == other.name and self.label == other.label
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.label))

    def __repr__(self) -> str:
        return f"Entity(name={self.name!r}, label={self.label!r}, properties={self.properties!r})"

# -------------------------------------------
# Relationship model
# -------------------------------------------
class Relationship(BaseModelWithConfig):
    startEntity: Entity = Field(default_factory=Entity)
    endEntity:   Entity = Field(default_factory=Entity)
    name:        str    = ""
    properties:  RelationshipProperties = Field(default_factory=RelationshipProperties)

    def process(self) -> "Relationship":
        self.name = LABEL_PATTERN.sub("_", self.name).replace("&", "and")
        return self
    
    def add_timestamp(self, timestamps:Union[List[float], List[str]]) -> None:
        if isinstance(timestamps[0], str):
            timestamps = [dateparser.parse(ts).timestamp() for ts in timestamps]
            self.properties.timestamps.extend(timestamps)
        elif isinstance(timestamps[0], float):
            self.properties.timestamps.extend(timestamps)
        else:
            raise ValueError("Invalid timestamp format. Please provide a list of strings or a list of floats.")

    def __eq__(self, other) -> bool:
        """Checks equality without considering timestamps."""
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity
                    and self.name == other.name)
        return False
    
    def __eq_with_timestamps__(self, other) -> bool:
        """Checks equality considering timestamps as a major differentiator."""
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity
                    and self.name == other.name
                    and set(self.properties.timestamps) == set(other.properties.timestamps))  # Timestamps must match exactly
        return False

    def __hash__(self, include_timestamps:bool=False) -> int:
        if include_timestamps:
            return hash((self.name, self.startEntity, self.endEntity, frozenset(self.properties.timestamps)))
        return hash((self.name, self.startEntity, self.endEntity))

    def __repr__(self) -> str:
        return (f"Relationship(name={self.name!r}, "
                f"startEntity={self.startEntity!r}, "
                f"endEntity={self.endEntity!r}, "
                f"properties={self.properties!r})")

# -------------------------------------------
# KnowledgeGraph model
# -------------------------------------------
class KnowledgeGraph(BaseModelWithConfig):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

    def remove_duplicates_entities(self) -> None:
        self.entities = list(set(self.entities))

    def remove_duplicates_relationships(self, include_timestamps:bool=False) -> None:
        """Removes duplicate relationships using hashing. 
        If `use_timestamps` is True, includes timestamps in hashing."""

        unique_relationships = {rel.__hash__(include_timestamps=include_timestamps): rel for rel in self.relationships}
        self.relationships = list(unique_relationships.values())

    async def embed_entities(self,
                             embeddings_function: Callable[[List[str]], np.ndarray],
                             entity_name_weight: float = 0.6,
                             entity_label_weight: float = 0.4) -> None:
        self.remove_duplicates_entities()
        self.entities = list(map(lambda e: e.process(), self.entities))

        labels = [e.label for e in self.entities]
        names  = [e.name for e in self.entities]

        label_embeddings = await embeddings_function(labels)
        name_embeddings  = await embeddings_function(names)

        for e, le, ne in zip(self.entities, label_embeddings, name_embeddings):
            e.properties.embeddings = entity_label_weight * le + entity_name_weight * ne

    async def embed_relationships(self,
                                  embeddings_function: Callable[[List[str]], np.ndarray]) -> None:
        self.remove_duplicates_relationships()
        self.relationships = list(map(lambda r: r.process(), self.relationships))

        names = [r.name for r in self.relationships]
        rel_embeddings = await embeddings_function(names)

        for r, emb in zip(self.relationships, rel_embeddings):
            r.properties.embeddings = emb

    def get_entity(self, other_entity: Entity) -> Entity:
        """Finds and returns an entity using a fast dictionary lookup."""
        other_entity = other_entity.process()
        entity_dict = {hash(e): e for e in self.entities}  # O(n) preprocessing, O(1) lookup
        return entity_dict.get(hash(other_entity))

    def get_relationship(self, other_relationship: Relationship, include_timestamps: bool = False) -> Relationship:
        """Finds and returns a relationship using a fast dictionary lookup."""
        other_relationship = other_relationship.process()
        relationship_dict = {
            rel.__hash__(include_timestamps=include_timestamps): rel for rel in self.relationships
        }
        return relationship_dict.get(other_relationship.__hash__(include_timestamps=include_timestamps))
    
    def add_timestamps_to_relationships(self, timestamps:Union[List[float], List[str]]) -> None:
        """Adds timestamps to relationships."""
        for rel in self.relationships:
            rel.add_timestamp(timestamps)
    
    def add_sources_to_relationships(self, source:str) -> None:
        """Adds sources to relationships."""
        if self.relationships:
            for rel in self.relationships:
                rel.properties.source = source

    def find_isolated_entities(self) -> List[Entity]:
        related_entities = set(r.startEntity for r in self.relationships) | \
                           set(r.endEntity   for r in self.relationships)
        return [ent for ent in self.entities if ent not in related_entities]

    def __repr__(self) -> str:
        return (f"KnowledgeGraph("
                f"entities={self.entities!r}, "
                f"relationships={self.relationships!r})")
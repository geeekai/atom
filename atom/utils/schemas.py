from typing import List, Optional
from pydantic import BaseModel, Field



# ----------------- Company Website  ---------------- # 
class FeatureDescription(BaseModel):
    feature_name : str = Field(description="the name of the product or service provided by the company")
    feature_caracteristics : List = Field(description="caracteristics of this product of service")
    
class JobDescription(BaseModel):
    job_title : str = Field(description="the title of the opening job")
    job_caracteristics : List = Field(description="caracteristics of opening job")

class EmployeeDescription(BaseModel):
    employee_name : str = Field(description="the name of the employee")
    employee_function : List = Field(description="the function of the employee")
    
class InformationRetriever(BaseModel):
    company_name: str = Field(description="name of the company")
    products_or_services : FeatureDescription = Field(description="the features of the products or services provided by the company")
    jobs : JobDescription = Field(description="the opening jobs proposed by the company")
    clients : List = Field(description= "the clients of the company")
    team : EmployeeDescription = Field(description="the team of the company")
    
# ----------------- Scientific Article ---------------- # 
class Author(BaseModel):
    name : str=Field(description="The name of the article's author. The right name should be near the article's title.")
    affiliation: str = Field(description="The affiliation of the article's author")
    
class ArticleDescription(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="Brief summary of the article's abstract")

    
class Article(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="The article's abstract")
    key_findings:str = Field(description="The key findings of the article")
    limitation_of_sota : str=Field(description="limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")
    
# ---------------- Entities & Relationships Extraction --------------------------- #

class Entity(BaseModel):
    label: str = Field(
        description=(
            "The semantic category of the entity (e.g., 'Person', 'Event', 'Location', 'Methodology', 'Position'). "
            "Use 'Relationship' objects if the concept is inherently relational or verbal (e.g., 'plans'). "
            "Prefer consistent, single-word categories where possible (e.g., 'Person', not 'Person_Entity')."
            "Do not extract Date Entities as they will be integrated in the relation."
        )
    )
    name: str = Field(
        description=(
            "The unique name or title identifying this entity, representing exactly one concept. "
            "For example, 'Yassir', 'CEO', or 'X'. Avoid combining multiple concepts (e.g., 'CEO of X'), "
            "since linking them should be done via Relationship objects. "
            "Verbs or multi-concept phrases (e.g., 'plans an escape') typically belong in Relationship objects."
            "Do not extract Date Entities as they will be integrated in the relation."
        )
    )
"""     named_entity: bool = Field(
        description="A named entity is a specific proper noun that refers to a unique person, organization, location, event or a product (e.g., ‘New York’, ‘Apple Inc.’, ‘Albert Einstein’)"
    ) """

class EntitiesExtractor(BaseModel):
    entities: List[Entity] = Field(
        description=(
            "A list of distinct entities extracted from text, each encoding exactly one concept "
            "(e.g., Person('Yassir'), Position('CEO'), Organization('X')). "
            "If verbs or actions appear, place them in a Relationship object rather than as an Entity. "
            "For instance, 'haira plans an escape' should yield separate Entities for Person('Haira'), Event('Escape'), "
            "and possibly a Relationship('Haira' -> 'plans' -> 'Escape')."
        )
    )

class Relationship(BaseModel):
    startNode: Entity = Field(
        description=(
            "The 'subject' or source entity of this relationship, which must appear in the EntitiesExtractor. "
        )
    )
    endNode: Entity = Field(
        description=(
            "The 'object' or target entity of this relationship, which must also appear in the EntitiesExtractor. "
        )
    )
    name: str = Field(
        description=(
            "A single, canonical predicate capturing how the startNode and endNode relate (e.g., 'is_CEO', "
            "'holds_position', 'located_in'). Avoid compound verbs (e.g., 'plans_and_executes'). "
            "If the text implies negation (e.g., 'no longer CEO'), still use the affirmative form (e.g., 'is_CEO') "
            "and rely on 't_invalid' for the end date. AVOID relations names as prepositions 'of', 'in' or similar."
        )
    )
    t_valid: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "A time or interval indicating when this relationship begins or is active. "
            "For example, if 'Yassir was CEO from 2023 to 2025', then t_valid='2023'. "
            "This can be a single year, a date (e.g., '2023-01-01' or '2023'). "
            "Leave it [] if not specified."
        )
    )
    t_invalid: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "A time or interval indicating when this relationship ceases to hold. "
            "For example, if 'Yassir left his position in 2025', then t_invalid='2025'. "
            "Use this field to capture any 'end action' (e.g., leaving a job, ending a marriage), "
            "while keeping the relationship name in a canonical positive form (e.g., 'is_CEO'). "
            "Leave it [] if no end date/time is given."
        )
    )

    
class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship] = Field(description="Based on the provided entities and context, identify the predicates that define relationships between these entities. The predicates should be chosen with precision to accurately reflect the expressed relationships.")
    
    
# ---------------------------- CV ------------------------------------- #

class WorkExperience(BaseModel):
    title: str
    company: str
    location: str
    start_date: str
    end_date: str
    responsibilities: List[str]

class Education(BaseModel):
    degree: str
    institution: str
    location: str
    start_date: str
    end_date: str
    coursework: Optional[List[str]]

class CV(BaseModel):
    name: str = Field(..., description="The name of the profile")
    phone_number: str = Field(..., description="The phone number of the profile")
    email: Optional[str] = Field(None, description="The email address of the profile")
    linkedin: Optional[str] = Field(None, description="The LinkedIn profile URL")
    summary: str = Field(..., description="A summary or professional profile")
    work_experience: List[WorkExperience] = Field(..., description="List of work experiences")
    education: List[Education] = Field(..., description="List of educational qualifications")
    skills: List[str] = Field(..., description="List of skills")
    certifications: Optional[List[str]] = Field(None, description="List of certifications")
    languages: Optional[List[str]] = Field(None, description="List of languages known")
    volunteer_work: Optional[List[str]] = Field(None, description="List of volunteer work experiences")
    
# ---------------------------- News ------------------------------------- #

class Fact(BaseModel):
    statement: str = Field(description="A factual statement mentioned in the news article")
    source: Optional[str] = Field(description="The source of the fact, if mentioned")
    relevance: Optional[str] = Field(description="The relevance or importance of the fact to the overall article")

class ArticleContent(BaseModel):
    headline: str = Field(description="The title or headline of the news article")
    subheading: Optional[str] = Field(description="The subheading or supporting title of the article")
    facts: List[Fact] = Field(description="List of factual statements covered in the article")
    keywords: List[str] = Field(description="List of keywords or topics covered in the article")
    publication_date: str = Field(description="The publication date of the article")
    location: Optional[str] = Field(description="The location relevant to the article")

class NewsArticle(BaseModel):
    title: str = Field(description="The title or headline of the news article")
    author: Author = Field(description="The author of the article")
    content: ArticleContent = Field(description="The body and details of the news article")

# ---------------------------- Novels ------------------------------------- #
class Character(BaseModel):
    name: str = Field(description="The name of the character in the novel")
    role: str = Field(description="The role of the character in the story, e.g., protagonist, antagonist, etc.")
    description: Optional[str] = Field(description="A brief description of the character's background or traits")

class PlotPoint(BaseModel):
    chapter_number: int = Field(description="The chapter number where this event occurs")
    event: str = Field(description="A significant event or plot point that occurs in the chapter")

class Novel(BaseModel):
    title: str = Field(description="The title of the novel")
    author: str = Field(description="The author of the novel")
    genre: str = Field(description="The genre of the novel")
    characters: List[Character] = Field(description="The list of main characters in the novel")
    plot_summary: str = Field(description="A brief summary of the overall plot")
    key_plot_points: List[PlotPoint] = Field(description="Key plot points or events in the novel")
    themes: Optional[List[str]] = Field(description="Main themes explored in the novel, e.g., love, revenge, etc.")

# ------------------------------- Temporal Atomic Facts Extraction ---------------------------------------- #

class Factoid(BaseModel):
    phrase: list[str] = Field(
        description="""
        **Guidelines for Generating Temporal Factoids**:

        1. **Atomic Factoids**:
           - Convert compound or complex sentences into short, single-fact statements.
           - Each Factoid must contain exactly one piece of information or relationship.
           - Ensure that each Factoid is expressed directly and concisely, without redundancies or duplicating the same information across multiple statements.
            For example, Unsupervised learning is dedicated to discovering intrinsic patterns in unlabeled datasets becomes "Unsupervised learning discovers patterns in unlabeled data."
        2. **Decontextualization**:
           - Replace pronouns (e.g., "it," "he," "they") with the full entity name or a clarifying noun phrase.
           - Include any necessary modifiers so that each Factoid is understandable in isolation.

        3. **Temporal Context**:
           - If the text contains explicit time references (e.g., "in 1995," "next Tuesday," "during the 20th century"), 
             include them in the Factoid so it is clear when the statement was or will be true.
           - Position the time reference in a natural place within the Factoid.
           - If a sentence references multiple distinct times, split it into separate Factoids as needed.

        4. **Accuracy & Completeness**:
           - Preserve the original meaning without combining multiple facts into a single statement.
           - Avoid adding details not present in the source text.

        5.  **End Actions**:
           - If the text indicates the end of a role or an action (for example, someone leaving a position),
             be explicit about the role/action and the time it ended.
        
        **Redundancies**:
        - Eliminate redundancies by simplifying phrases (e.g., convert "the method is crucial for maintaining X" into "the method maintains X").

        **Example**:
        On June 18, 2024, Real Madrid won the Champions League final with a 2-1 victory. Following the triumph, fans of Real Madrid celebrated the Champions League victory across the city.
        -Real Madrid won the Champions League final on June 18, 2024.
        -The final Champions League final ended with a 2-1 victory for Real Madrid on June 18, 2024.
        -Fans of Real Madrid celebrated the victory of Champions League final across the city on June 18, 2024.
        """
    )
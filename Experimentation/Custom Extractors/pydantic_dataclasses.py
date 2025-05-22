from pydantic import BaseModel, Field, HttpUrl, conint
from typing import List, Optional
from datetime import date

class Author(BaseModel):
    name: str = Field(..., description="Name of the author")
    organization: Optional[str] = Field(None, description="Organization affiliated with the author")
    contact_info: Optional[str] = Field(None, description="Contact information for the author")

class Section(BaseModel):
    title: str = Field(..., description="Title of the section or chapter")
    start_page: conint(ge=1) = Field(..., description="Starting page number of the section")
    end_page: conint(ge=1) = Field(..., description="Ending page number of the section")
    summary: Optional[str] = Field(None, description="Brief summary of the section's content")

class Specification(BaseModel):
    type: str = Field(..., description="Type of specification")
    details: str = Field(..., description="Details about the specification")

class MarineEngineeringManual(BaseModel):
    title: str = Field(..., description="The title of the manual")
    authors: List[Author] = Field(..., description="List of authors responsible for the manual")
    publication_date: date = Field(..., description="The publication date of the manual")
    edition: Optional[str] = Field(None, description="The edition of the manual, if applicable")
    isbn: Optional[str] = Field(None, description="The ISBN number of the manual, if available")
    url: Optional[HttpUrl] = Field(None, description="A URL where the manual can be accessed, if available")
    ship_types: List[str] = Field(..., description="Types of ships the manual is applicable to")
    sections: List[Section] = Field(..., description="List of major sections or chapters in the manual")
    pages: conint(ge=1) = Field(..., description="Total number of pages in the manual")
    language: str = Field(..., description="The language in which the manual is written")
    classification_code: Optional[str] = Field(None, description="Classification code if the manual is categorized")
    keywords: List[str] = Field(..., description="List of relevant keywords for searching or categorization")
    description: Optional[str] = Field(None, description="A brief description or summary of the manual")
    publisher: Optional[str] = Field(None, description="Name of the publisher")
    publication_location: Optional[str] = Field(None, description="Location where the manual was published")
    revision_history: Optional[List[str]] = Field(None, description="History of revisions made to the manual")
    diagrams_url: Optional[HttpUrl] = Field(None, description="URL to access diagrams related to the manual")
    references: Optional[List[str]] = Field(None, description="List of references used in the manual")
    specifications: Optional[List[Specification]] = Field(None, description="Technical specifications mentioned in the manual")
    appendices: Optional[List[str]] = Field(None, description="List of appendices included in the manual")
    notes: Optional[str] = Field(None, description="Any additional notes or comments about the manual")
    safety_information: Optional[str] = Field(None, description="Safety information related to the content of the manual")
    compliance_standards: Optional[List[str]] = Field(None, description="List of compliance standards referenced in the manual")
    review_date: Optional[date] = Field(None, description="The date when the manual was last reviewed or updated")
    related_manuals: Optional[List[str]] = Field(None, description="Titles of other manuals related to this one")

# Example Usage
    # manual = MarineEngineeringManual(
    #     title="Advanced Marine Engineering Practices",
    #     authors=[Author(name="John Doe", organization="Marine Tech Inc.", contact_info="john.doe@example.com")],
    #     publication_date=date(2021, 7, 22),
    #     edition="2nd Edition",
    #     isbn="978-3-16-148410-0",
    #     url="http://example.com/advanced-manual",
    #     ship_types=["Tanker", "Cargo", "Cruise Ship"],
    #     sections=[
    #         Section(title="Introduction", start_page=1, end_page=10, summary="Overview of marine engineering practices"),
    #         Section(title="Engines", start_page=11, end_page=50, summary="Detailed analysis of marine engines")
    #     ],
    #     pages=500,
    #     language="English",
    #     classification_code="620.16",
    #     keywords=["marine", "engineering", "safety", "maintenance", "advanced"],
    #     description="A comprehensive guide on advanced marine engineering practices.",
    #     publisher="Marine Publishers",
    #     publication_location="New York, USA",
    #     revision_history=["1st Edition - 2018", "2nd Edition - 2021"],
    #     diagrams_url="http://example.com/diagrams",
    #     references=["Marine Standards 2020", "Safety Protocols 2019"],
    #     specifications=[
    #         Specification(type="Engine Specifications", details="Details about various types of marine engines"),
    #         Specification(type="Safety Equipment", details="List and description of safety equipment")
    #     ],
    #     appendices=["Appendix A - Glossary", "Appendix B - Index"],
    #     notes="This manual is intended for advanced marine engineers.",
    #     safety_information="Ensure proper safety protocols are followed when using this manual.",
    #     compliance_standards=["ISO 9001", "IMO Safety Standards"],
    #     review_date=date(2023, 6, 1),
    #     related_manuals=["Marine Engineering Basics", "Ship Maintenance Guide"]
    # )

    # print(manual)

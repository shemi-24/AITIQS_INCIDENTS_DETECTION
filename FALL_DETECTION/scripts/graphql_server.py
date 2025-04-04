import strawberry
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.asgi import GraphQL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a simple Query type (required for GraphQL schema)
@strawberry.type
class Query:
    hello: str = "Hello, GraphQL!"

# Define GraphQL Mutation
@strawberry.type
class Mutation:
    @strawberry.mutation
    def report_fall(self, person_name: str, image_url: str) -> str:
        logger.info(f"⚠️ Fall detected! Person: {person_name}, Image: {image_url}")
        return f"Fall event recorded for {person_name}"

# Create GraphQL Schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create FastAPI App
app = FastAPI()

# Add CORS (Allows frontend apps to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GraphQL route
app.add_route("/graphql", GraphQL(schema))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

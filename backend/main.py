"""FastAPI main entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import linear_regression, logistic_regression, knn, decision_tree, random_forest

app = FastAPI(
    title="DataScienceApp API",
    description="Production-grade ML Education backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Must be False if using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(linear_regression.router)
app.include_router(logistic_regression.router)
app.include_router(knn.router)
app.include_router(decision_tree.router)
app.include_router(random_forest.router)


@app.get("/")
def root():
    return {"message": "DataScienceApp API is running", "docs": "/docs"}

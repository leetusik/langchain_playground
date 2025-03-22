"""Test script to verify all package imports work together without conflicts."""

import importlib
import sys


def test_imports():
    packages = [
        # Core
        "pandas",
        "numpy",
        "dotenv",
        # LangChain
        "langchain",
        "langchain.text_splitter",
        "langchain_community.vectorstores",
        "langchain_core.documents",
        "langchain_core.embeddings",
        "langchain_openai",
        "langchain_pinecone",
        # Vector DB
        "pinecone",
        # API
        "openai",
    ]

    success = True
    for package in packages:
        try:
            module = importlib.import_module(package)
            print(
                f"✅ Successfully imported {package} (version: {getattr(module, '__version__', 'unknown')})"
            )
        except ImportError as e:
            print(f"❌ Failed to import {package}: {str(e)}")
            success = False

    if success:
        print("\n🎉 All imports successful! No apparent version conflicts.")
    else:
        print("\n❗ Some imports failed. Review the output for details.")

    return success


if __name__ == "__main__":
    sys.exit(0 if test_imports() else 1)

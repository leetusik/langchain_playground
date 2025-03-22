import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .services.chain import answer_chain

logger = logging.getLogger(__name__)


def chat_page(request):
    """
    Renders a simple chat interface for testing the RAG system
    """
    return render(request, "chatbot/chat.html")


@api_view(["POST"])
def chat_api(request):
    """
    API endpoint for the chatbot that accepts:
    {
        "question": "Your question here",
        "chat_history": [{"human": "previous question", "ai": "previous answer"}, ...]
    }
    """
    try:
        data = request.data
        question = data.get("question", "")
        chat_history = data.get("chat_history", [])

        if not question:
            return Response(
                {"error": "Question is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Log the input
        logger.info(f"Received question: {question}")
        if chat_history:
            logger.info(f"With chat history of {len(chat_history)} messages")

        # Prepare input for the chain
        chain_input = {"question": question, "chat_history": chat_history}

        # Run the chain and get the response
        response = answer_chain.invoke(chain_input)

        # Return the response
        return Response(
            {
                "answer": response,
                "question": question,
            }
        )

    except Exception as e:
        logger.error(f"Error in chat_api: {str(e)}")
        return Response(
            {"error": f"An error occurred: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@csrf_exempt
def chat_simple(request):
    """
    Simple JSON endpoint for the chatbot without DRF dependency
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests allowed"}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question", "")
        chat_history = data.get("chat_history", [])

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        # Prepare input for the chain
        chain_input = {"question": question, "chat_history": chat_history}

        # Run the chain and get the response
        response = answer_chain.invoke(chain_input)

        # Return the response
        return JsonResponse(
            {
                "answer": response,
                "question": question,
            }
        )

    except Exception as e:
        logger.error(f"Error in chat_simple: {str(e)}")
        return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

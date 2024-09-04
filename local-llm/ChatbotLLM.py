import os
import threading
import tkinter as tk
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

class Chatbot():
    """
    Chatbot class takes in user question and responds with the output.
    """
    class GraphState(TypedDict):
        """
        GraphState class stores important information like the user's question and answer.
        """
        question: str
        query: str
        generation: str
        documents: List[str]

    class NewRetriever(VectorStoreRetriever):
        """
        NewRetriever class wraps around the VectorStoreRetriever from Langchain but allows searching through the entire database instead of a random subset.
        """
        def invoke(self, query):
            return super().invoke(query)[:Chatbot.NUM_DOCUMENTS]

    NUM_DOCUMENTS = 5
    LLM_NAME = 'llama3'
    EMBEDDINGS = HuggingFaceEmbeddings()

    # Turns the user's question into a query that is more suited for searching the database for relevant questions.
    QUERY_CREATION_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Given the question and chat history below, 
        generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the question: {question}
        Here is the chat history: {history}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
        """,
        input_variables=['question', 'history']
    )

    # Summarizes all previous chat messages to reduce the size of the chat history.
    SUMMARIZE_MESSAGES_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Distill the following chat messages into a single summary message. 
        Include as many specific details as you can.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the chat history: {history}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
        """,
        input_variables=['history']
    )

    # Checks if the documents retrieved from the database query are relevant to answering the user's question.
    RETRIEVAL_GRADER_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        Here is the chat history: {history}
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Here is the retrieved document: \n\n {document} \n\n 
        Here is the user question: {question} 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>  
        """,
        input_variables=['question', 'document', 'history'],
    )

    # Uses the question and returned documents to generate a response.
    GENERATE_ANSWER_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context and the chat history to answer the question. If you don't know the answer, just say that you don't know. 
        Do not provide a link to an outside source if an explanation can be provided. Use three sentences maximum and keep the answer concise. 
        You must provide the FAQ number or document for the questions you reference for the answer which is given in the document metadata tag 'document'
        with format 'FAQ #_' or a sentence.
        Here is the chat history: {history}
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>  
        """,
        input_variables=['question', 'context', 'history']
    )

    # Generates a response to the question without using any documents if none of them are relevant to the question.
    GENERATE_WITHOUT_DOCUMENTS_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use your pretrained knowledge and the chat history to answer the question. Use three sentences maximum and keep the answer concise. 
        Here is the chat history: {history}
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Question: {question} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>  
        """,
        input_variables=['question', 'history']
    )

    # Checks if the generated answer contains factual information without hallucinations.
    HALLUCINATION_GRADER_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|> 
        Here are the facts: 
        \n ------- \n 
        {documents} 
        \n ------- \n 
        Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
        """,
        input_variables=['documents', 'generation']
    )

    # Checks if the generated answer answers the question being asked.
    ANSWER_GRADER_PROMPT = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Be lenient and allow answers that provide some information about the question, 
        even if it does not completely know. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer: 
        \n ------- \n 
        {generation} 
        \n ------- \n 
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|> 
        """,
        input_variables=['generation', 'question']
    )

    def __init__(self):
        """
        Initializes the chatbot class.
        """

        # Loads in the database and creates the retriever to find documents
        self.vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=self.EMBEDDINGS, collection_name="faqs")
        self.VECTORSTORE_SIZE = len(self.vectorstore.get()['ids'])
        self.retriever = self.NewRetriever(vectorstore=self.vectorstore, search_type='similarity', search_kwargs={'k': self.VECTORSTORE_SIZE})
        self.history = []

        # Set up the Ollama LLM for each step in the pipeline
        llm = ChatOllama(model=self.LLM_NAME, format='json', temperature=0)
        self.retrieval_grader = self.RETRIEVAL_GRADER_PROMPT | llm | JsonOutputParser()
        self.hallucination_grader = self.HALLUCINATION_GRADER_PROMPT | llm | JsonOutputParser()
        self.answer_grader = self.ANSWER_GRADER_PROMPT | llm | JsonOutputParser()
        llm = ChatOllama(model=self.LLM_NAME, temperature=0)
        self.query_creator = self.QUERY_CREATION_PROMPT | llm | StrOutputParser()
        self.generator = self.GENERATE_ANSWER_PROMPT | llm | StrOutputParser()
        self.generator_no_docs = self.GENERATE_WITHOUT_DOCUMENTS_PROMPT | llm | StrOutputParser()

        # Create workflow to process the question
        workflow = StateGraph(self.GraphState)
        workflow.add_node('reformat', self.reformat)
        workflow.add_node('retrieve', self.retrieve)
        workflow.add_node('grade_documents', self.grade_documents)
        workflow.add_node('generate', self.generate)

        # Currently the node to validate the answer is not being used to improve response time
        # workflow.add_node('generate_no_docs', self.generate_no_docs)

        workflow.set_entry_point('reformat')
        workflow.add_edge('reformat', 'retrieve')
        workflow.add_edge('retrieve', 'grade_documents')
        workflow.add_edge('grade_documents', 'generate')

        # Currently the node to validate the answer is not being used to improve response time
        # workflow.add_conditional_edges(
        #     'generate',
        #     self.grade_generation,
        #     {
        #         'useful': END,
        #         'not useful': 'generate_no_docs'
        #     }
        # )
        # workflow.add_edge('generate_no_docs', END)

        workflow.add_edge('generate', END)

        self.graph = workflow.compile()
    
    def summarize_messages(self):
        """
        Reduces the size of the chat history to improve performance by summarizing all messages into one message.
        """
        llm = ChatOllama(model=self.LLM_NAME, temperature=0)
        summarizer = self.SUMMARIZE_MESSAGES_PROMPT | llm | StrOutputParser()
        summary = summarizer.invoke({'history': self.history})
        self.history = [AIMessage(content=summary)]
        return 

    def reformat(self, state):
        """
        Reformats question into a query that can better find documents in the database.
        """
        print('---REFORMAT QUESTION---')
        question = state['question']

        query = self.query_creator.invoke({'question': question, 'history': self.history})
        return {'query': query, 'question': question}

    def retrieve(self, state):
        """
        Uses the query to match documents in the database that are similar to the question.
        """
        print('---RETRIEVE---')
        query = state['query']

        documents = self.retriever.invoke(query)
        return {'documents': documents, 'query': query}

    def grade_documents(self, state):
        """
        Determine which documents are useful for answering the question.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state['question']
        documents = state['documents']

        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({'question': question, 'document': d.page_content, 'history': self.history})
            grade = score['score']
            if grade.lower() == 'yes':
                print('---GRADE: DOCUMENT RELEVANT---')
                filtered_docs.append(d)
            else:
                print('---GRADE: DOCUMENT NOT RELEVANT---')

        return {'documents': filtered_docs, 'question': question}
        
    def generate(self, state):
        """
        Generate the answer to the question.
        """
        print('---GENERATE---')
        question = state['question']
        documents = state['documents']

        generation = self.generator.invoke({'context': documents, 'question': question, 'history': self.history})
        return {'documents': documents, 'question': question, 'generation': generation}
        
    def generate_no_docs(self, state):
        """
        Generate an answer without using any documents.
        """
        print('---GENERATE WITHOUT DOCUMENTS---')
        question = state['question']

        generation = self.generator_no_docs.invoke({'question': question, 'history': self.history})
        return {'documents': [], 'question': question, 'generation': generation}
        
    def grade_generation(self, state):
        """
        Check the quality of the answer that is being outputted.
        """
        print('---CHECK HALLUCINATIONS---')
        question = state['question']
        documents = state['documents']
        generation = state['generation']

        score = self.hallucination_grader.invoke({'documents': documents, 'generation': generation, 'history': self.history})
        grade = score['score']

        if grade.lower() == 'yes':
            print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')

            print('---GRADE GENERATION VS QUESTION---')
            score = self.answer_grader.invoke({'question': question, 'generation': generation, 'history': self.history})
            grade = score['score']

            if grade.lower() == 'yes':
                print('---DECISION: GENERATION ADDRESSES QUESTION---')
                return 'useful'
            else:
                print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
                return 'not useful'
        else:
            print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---')
            return 'not useful'
            
    def run(self, question):
        """
        Takes in the question and follows the pipeline to generate an answer.
        """
        inputs = {'question': question}
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"Finished Running '{key}':")
        self.history.extend([HumanMessage(content=question), AIMessage(content=value['generation'])])
        return value['generation']
    
class ChatbotGUI:
    """
    ChatbotGUI class create a text box for asking questions.
    """
    def __init__(self, root):
        """
        Creates text boxes for question and answer.
        """
        self.bot = Chatbot()
        self.root = root
        self.root.title("Chatbot")
        
        self.question_label = tk.Label(root, text="Ask a question:")
        self.question_label.pack()
        
        self.question_entry = tk.Text(root, height=5, width=100)
        self.question_entry.pack()
        
        self.answer_label = tk.Label(root, text="Answer:")
        self.answer_label.pack()
        
        self.answer_text = tk.Text(root, height=10, width=100)
        self.answer_text.pack()
        
        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.pack()
        
    def ask_question(self):
        """
        Takes in the user's question.
        """
        question = self.question_entry.get("1.0", tk.END).strip()
        self.answer_text.delete("1.0", tk.END)
        thread = threading.Thread(target=self.generate_response, args=(question,))
        thread.start()
        
    def generate_response(self, question):
        """
        Outputs the response to the user's question
        """
        answer = self.bot.run(question)
        self.answer_text.insert(tk.END, answer)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()
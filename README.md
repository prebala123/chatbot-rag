# Chatbot

OpsMx provides continuous delivery solutions to its customers using Spinnaker and is one of the top contributors to open source Spinnaker. OpsMx has built a body of knowledge distributed in various documents as FAQs, Blogs and Customer Support issues that encapsulate knowledge of the systems over the years. The knowledge base includes issues and solutions across multiple cloud environments, Spinnaker specific configurations and solutions. With the data being distributed and not easily accessible for a given problem, we wanted to use LLMs to make the information accessible as self service for its support personnel as well as customers to provide better solutions for troubleshooting and configuration options. We hope to make a chatbot that can efficiently use historical data on all the previous incidents as well as other documents to come to a solution to a problem, to improve response time and quality.

# Overview

Our LLM required a good base model to formulate responses with pre trained knowledge as well as the ability to access our data to search for the correct solutions to the customers’ problems. We followed these steps to build our model, give it access to our data stores, and publish it for general use.

## Selecting the model

Our first step was determining the base LLM model to use and how we were going to train it. We tried two separate methods for this and started with finetuning but eventually went with a RAG approach.

Finetuning is about choosing a subset of the weights of a base LLM and changing them based on the training data. This would allow our company’s data to be baked into the model’s knowledge and it would be used for any response. We chose the base model of llama3 with 7 billion parameters from Ollama which would make local training easy. However we later abandoned this method to go with a RAG approach.

Using RAG, Retrieval Augmented Generation, we create a database of our training data instead of actually putting it in the model. We create vector embeddings for every document such that similar documents will have similar values. When the user asks a question, we can find the vector embedding of the question and use it to find documents in our database that have similar embeddings, which provides context for our answer. For this method we chose a gemini model from vertex ai for fast and high quality responses.

## Data cleaning

Our data was in the form of FAQs in google docs and previous incident data in google sheets. We used the google drive API to access the data and separate them into separate question and answer pairs. For the FAQ data, we simply parsed the document and extracted the question and answer data. For the incidents, we tagged the problem description as the question and the implemented solution as the answer. We created hundreds of documents, each containing a question and answer pair to train the model. Our created documents were stored as a list of dictionaries containing the title of the problem, the question the user asked, and the expected answer. We used pandas DataFrame methods to get the data ready.

## Training the model

We first tried training the llama3 model using finetuning on a local machine. We used a PyTorch and Unsloth environment to choose a subset of model weights and alter them with our documents. Each piece of training data contained one question and answer pair from a document as well as a prompt explaining how to format the answer based on the question. We ran the training loop for several hours but were not happy with the results.

We then moved to google cloud and used its features to create our RAG implementation using Gemini as the LLM and Langchain as the framework. In vertex ai, we created a colab notebook to take in our training data consisting of questions and answers. We created a vector search index to store the embeddings of each document, as well as an endpoint to retrieve relevant documents to get the right questions and answers. 

We loaded the documents into this data store by first creating a vector store object to do the embeddings. We also created a retriever to match a query with documents with similar embeddings. From there we uploaded a list of document objects containing our information,  which are the needed type for the vectorstore.

We then created a pipeline in Langchain to retrieve the 4 most relevant documents to the user’s question. A prompt would be created writing in all these documents as well as an instruction to answer the user’s question based on the information contained in the context. The outputs using this method were much better than the previous method.

## Pushing to production

We tested the model with slight changes to certain parameters like number of documents retrieved or model temperature to find the best possible output. We got feedback from internal teams using their example questions to ensure high quality outputs for the model. To push this to production, we created a docker container with our model’s pipeline and all the necessary packages. Then we uploaded it to vertex ai’s model registry to host it for teams to use.

# Training Methods

## Finetuning

Our first idea was baking the knowledge into the weights of the LLM so that it could be used easily by just calling the LLM. This method is a lot easier to deploy and use, since only the model weights are needed for running it. It can also be trained locally saving on cost or on the cloud for better speed, giving more options. Since the data is already embedded in the model, prompts are very simple and easy for anyone to use.

However, we found that the results were more similar to the model’s base knowledge than the extra training data, meaning it wasn’t considering our examples as much. We wanted the model to prioritize the company’s data over what the model was trained on because it would be more relevant to the problem. In addition, training can take a lot of time and memory, making it hard for adjustments to be made, like adding more training data in the future.

## RAG

We used RAG to address the problem of the answers not being relevant to the training data which happened with finetuning. Since we are telling the LLM to generate the answers only based on similar answers, it is much more likely to be the result we want. This method saves a lot of training time since it is much faster to create a vector database than to tune an LLM. Since we are using just the base LLM, we can also change it easily between models like Gemini 1.5 Flash and Gemini 1.0 Pro, for different use cases.

However an issue with this is that we now have to maintain a vector database on google cloud which costs money. To keep the model running we now have to make sure the database and endpoints are all running rather than just having the weights. In addition, the database might not contain any documents relevant to the question if it is a completely new issue, making it unable to give an answer. Overall we found this method to be the better one because of its higher quality answers.


# Conclusion

Our final chatbot LLM responded with accurate information and was useful in answering questions about errors that could pop up while using OpsMx products. When creating the database of documents for the RAG pipeline, we embedded the documents with text-embedding-004 in vertex ai which provided good results in retrieving the documents. We could consistently retrieve documents that were relevant to the user’s question and could help create a useful answer. We also used the gemini-1.0-pro model as the base model because it was very good at summarizing the provided documents into an answer for the given question. By changing the model’s temperature to 0, we could make sure that the answers were more grounded to the documents and not hallucinating new information. We experimented with the number of documents to be included in the context for the LLM and decided on 4 as it was enough information while not exceeding the token limit or slowing performance down. Overall our setup gives satisfactory results for the questions and can be used in production to supplement our customer service representatives in their responses.

This repository contains the code for our local RAG implementation as well as the one on Google Cloud using Vertex AI.

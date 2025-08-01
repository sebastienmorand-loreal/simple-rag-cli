
==================================================
Reading: Full local open source lightweight simple RAG _ by Elias Tarnaras _ Medium.pdf
==================================================
# **_Full local open source lightweight_** **_simple RAG_**

_[Elias Tarnaras](https://medium.com/@elias.tarnaras?source=post_page---byline--a0a1de586209---------------------------------------)_ _Follow_ _18 min read · Jun 14, 2024_


_2_


## **_Introduction_**

In my journey to learn Machine Learning, NLP and LLMs, I very quickly came upon the term RAG (Retrieval Augmented Generation). RAG has been proposed and is being used as a solution to the hallucination problem of widely used LLMs. Hallucinations are answers from the chatbots that have no factual truth, but seem to be correct as they are very eloquently phrased.

While this problem is getting less and less of a problem with new, better trained and bigger models, RAG is still useful if we want an AI chatbot to give answers with proprietary information, or answers using less common information that the current widely used LLMs have not yet trained with.

So, I embarked on an exercise to create my own simple RAG system using only open source libraries and models. In my whole research online, I haven't yet seen an example of a full RAG system that does not use platforms like Ollama or LM Studio, or an abstraction library like LangChain or LlamaIndex. While these tools are extremely useful, some are cumbersome to install, and they 'steal' some of the developer freedom and control (e.g. using any model we choose). They also have a steep learning curve.

I wanted the whole thing to run locally, models, vector store etc., on my humble (for ML purposes) PC.


My PC configuration is:

OS: Windows 11
CPU: AMD Ryzen 9 5900X 12-Core Processor @ 3.70 GHz
RAM: 32.0 GB
GPU: Nvidia GeForce RTX 4070 (12 GB VRAM)

In this example, we will use:

- Llama Parse: For extracting markdown from PDF files
- ChromaDB: As a local Vector Store (database) with a custom embed function and model
- Hugging Face: For open source models and Python libraries

Models:
- w601sxs/b1ade-embed: As the embedding and retrieval model for ChromaDB
- microsoft/Phi-3-mini-128k-instruct: As the actual LLM

The concept of this simple RAG system will be like this.

Some terminology is needed here.


**Markdown**: Markdown is a lightweight markup language that we can use to [add formatting elements to plaintext text documents. Created by John Gruber in 2004, Markdown is now one of the world's most popular markup languages](https://daringfireball.net/projects/markdown/). [Here you can learn more about markdown](https://www.markdownguide.org/). Basically it is a very lightweight formatted text that supports tables and mathematical expressions. As it is very readable even when viewing it in a plain text editor, it is a good format for LLM context.

**Vector Store**: A database used to store embeddings and their relevant data. It can be searched for semantic relevance.

**Embedding**: Vector (array of numbers) representation of text. The only way LLMs can understand text is to somehow convert it to numbers. This way, the LLM can perform calculations to compute the relevance (similarity) between texts and give semantic meaning to text.

**LLM**: Large Language Model (Think Llama, Claude, ChatGPT models). These models can comprehend and generate human language text.

**RAG**: Retrieval Augmented Generation. A technique that uses information retrieval and passes this information as context to LLMs in order for the LLM to return a more factually correct answer.

## **_Our Example_**

We will ask for clarification of board game rules based on the game's official rulebook PDF file.

Board games have notoriously complex PDF files, and more often than not, a lot of convoluted and difficult to remember rules.

A RAG system could be a potential solution to this problem.

For our example we will ask our RAG system for the following board games:
- The Witcher: Old World
- Dune: Imperium — Uprising

If you want to follow the exact example, you can find the rulebooks PDF files using a simple internet search, or you can always use your own PDF files for the use you would like to try.

## **_Implementation_**

[Let's explore the code (you can find the full code in my GitHub)](https://github.com/eltarnny/simple-local-rag)

## **_Parse PDF file_**

[As explained in my previous article "Unlocking the Secrets of PDF Parsing: A Comparative Analysis of Python Libraries"](https://medium.com/@eltarny/unlocking-the-secrets-of-pdf-parsing-a-comparative-analysis-of-python-libraries-79064bf12174), I used Llama Parse in order to correctly (as much as possible) extract the text from complex PDF files.

Here is the code for reference:

First you need to install llama-index and llama-Parse. I am aware that we said that we will not use LlamaIndex, but this is the only part that we will use. Llama Parse offers 1000 pages per day for free. You will need to first [create a Llama Cloud account (or login using your Google/GitHub/Microsoft account) and then create a Llama Cloud API key (here is how)](https://cloud.llamaindex.ai/). Alternatively, you can use `pymupdf4llm` for a complete local solution, but it gets seriously confused with the column format of these PDF files and extracts the text in the wrong order.


To install use:

```
  pip install llama-index

  pip install llama-parse

```

We will need to import the following Python libraries:

```
  import nest_asyncio

  from llama_parse import LlamaParse

  from dotenv import load_dotenv
  import os

  nest_asyncio.apply()

  load_dotenv()

```

`nest_asyncio.apply()` and `load_dotenv()` are commands that according to Llama Parse need to be executed first. We also need `load_dotenv()` to get out API key from the `.env` file. We need to create this file and store our actual API key.

```
LLAMA_CLOUD_API_KEY=xxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

This is the rest of the parse code:

```
  # fname_no_ext = "TWOW_07_Rulebook_Main_(36p)_EN_preview_22-10-20-skompresowany"
  fname_no_ext = "DUNE_IMPERIUM_UPRISING_RULEBOOK"

  fname = fname_no_ext + ".pdf"

  # Initialize the parser with your API key and desired settings

  parser = LlamaParse(
  api_key=os.getenv('LLAMA_CLOUD_API_KEY'), # Replace with your actual API ke

  result_type="markdown", # "markdown" and "text" are available options
  num_workers=4, # Number of workers for parallel processing

  verbose=True, # Enable verbose output

  language="en" # Set the language (default is English)
  )

  # Load and parse a single PDF file asynchronously

  async_documents = await parser.aload_data(fname)

  # For the above 36 page pdf, it took 4m13s to parse

  if isinstance(async_documents, list) and async_documents:
     with open(fname_no_ext + '.md', 'w') as file:

  file.write(async_documents[0].text)

```

This code serves the PDF file to Llama Parse and saves the response as a markdown file. This process is very slow and can last from 5 to 10 or more minutes, depending on the size and complexity of the PDF file. You can set your own `fname_no_ext` here.

## **_Markdown to Vector Store_**

Now that we have the information as markdown text, we can convert it to embeddings and save into a local Vector Store (vector database). We will use [ChromaDB. You can read more about ChromaDB and how to use it in my previous article "YouTube History Semantic Search"](https://www.trychroma.com/).


To install ChromaDB use:


```
  pip install chromadb

```

Now we import the relevant libraries:

```
  from tools import *
  import chromadb

  from customEmbedFunction import TransformerEmbeddingFunction

```

where `tools` is a Python file with some of my helper functions and `customEmbedFunction` is the file that has the custom class `TransformerEmbeddingFunction`. This custom class will allow us to specifically select which model we will use in order to calculate the embeddings and do the similarity comparison.

By specifically selecting the embedding model ourselves, we can conduct tests on the similarity results in order to find the sweet spot of VRAM usage, execution time and quality of the returned results. In order to try several [models, I used the Massive Text Embedding Benchmark (MTEB) Leaderboard STS tab (Metric: Spearman correlation based on cosine similarity)](https://huggingface.co/spaces/mteb/leaderboard) from Hugging Face, and filtered out the big models.

As we can see, the best 'small' model for cosine similarity (a method to calculate semantic similarity of text pairs) is [b1ade-embed](https://huggingface.co/w601sxs/b1ade-embed). In my tests, this model proved both fast and capable.


First we need to get the text from our markdown file:

```
  # SET GAME NAME AND FILE HERE

  # game_name = 'The Witcher: Old World'

  game_name = 'Dune: Imperium - Uprising'

  # fname_no_ext = "TWOW_07_Rulebook_Main_(36p)_EN_preview_22-10-20-skompresowany"

  fname_no_ext = "DUNE_IMPERIUM_UPRISING_RULEBOOK"

  fname = fname_no_ext + '.md'

  content = read_text_file(fname)

```

Feel free to set your own `game_name` and `fname_no_ext` here.

Then we need to split the imported text into chunks. Embedding models have small context windows (how much text they can handle). I selected 300 words as my chunk size, which is a good value for the context window of `b1ade-embed`.

```
  chunks = split_text(content, max_words=300, overlap=0)

```

`split_text` is one of my helper functions in the `tools` file.

Then we set the model and create the ChromaDB vector store.

```
  embedder_name = "w601sxs/b1ade-embed"

  chroma_client = chromadb.PersistentClient(path="chroma")
  collection = chroma_client.get_or_create_collection(name='bg-rag', embedding_fun

```

Here we use the custom `TransformerEmbeddingFunction` in order to use our selected model. You can see the code of the custom function I got from [ChromaDB Cookbook in my GitHub repository](https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/). The `use_cuda=True` argument specifies that we will use our GPU for fast execution. This code will create a folder named `chroma` in which it will save the needed files of the `bg-rag` collection (our database).


Then we insert our text chunks in our vector store using the `upsert` (update or insert) command. With this command, if a chunk record already exists, it will be updated and not inserted again. Alternatively, we can use the `add` command to just add new records.

```
  num_chunks = len(chunks)
  num_digits = len(str(num_chunks))
  ids = [fname + "_" + str(i).zfill(num_digits) for i in range(num_chunks)]
  metadata = [{'filename': fname,

          'bg_name': bg_name,

          'bgg_id': bgg_id,
          'rules_forum_id': rules_forum_id,

```

```
          'general_forum_id': general_forum_id,
  } for _ in range(num_chunks)]

  collection.upsert(
  documents=chunks,

  metadatas=metadata,

  ids=ids

  )

Here, the `metadatas` are some data I save with the embeddings. To explain these specific metadata is out of the scope of this article. We also create our own unique ids in order to use them later, using the filename and the zero padded chunk number (e.g. `DUNE_IMPERIUM_UPRISING_RULEBOOK.md_03`).

## **_Simple RAG system_**

It's time to see our stored embeddings in action! For this part, apart from our vector store and our custom embedding function using the `b1ade-embed` model, we will use another model, the [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model, which is a very new model of the Phi3 model family from Microsoft.

This is a very small quantized (uses specially normalized smaller precision numbers for lower VRAM usage) model. It is a good choice, as it has a huge 128k context window and it is instruction trained. In fact, with only 12GB VRAM, we cannot even use the whole context window, but it gives us freedom to choose our input text size. RAG needs big context size (window), as we need to send all the relevant information along with our question to the model in order to get a factually correct answer.


[First we need to install PyTorch. From the link you can choose if you want to](https://pytorch.org/) install with or without CUDA (For explanation about CUDA see my previous [article "YouTube History Semantic Search"](https://medium.com/@eltarny/youtube-history-semantic-search-6cd9f20922a8)).


In my case, I used:

```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.or

```

Now we need to import the libraries:

```
  import torch

  import chromadb
  from customEmbedFunction import TransformerEmbeddingFunction

```

Then we get the vector store and the collection.

```
  chroma_client = chromadb.PersistentClient(path="chroma")

  embedder_name = "w601sxs/b1ade-embed"

  collection = chroma_client.get_or_create_collection(name='bg-rag', embedding_fun

```

Then we need to get our most relevant to our question chunks of texts from the vector store. During the tests, the name of the game in the question carried a lot of semantic meaning for the returned results. It was 'masking' the semantic meaning of the actual question. As a solution, we can use a trick to first we get a sample of results to find the game name using the most common game name of the returned results. Then we select new results just for this game and replace the name of the game in the question text with the generic text 'the game'.

```
  questions = []
  questions.append("For The Witcher: Old World, please tell me how to setup a 2-pl

  questions.append("In The Witcher: Old World, can I play poker and fight a player
  questions.append("In Dune: Imperium - Uprising, please explain in detail how spi

  number_of_results = 2 # This should be the perfect value for this model.

  extend_results_by = 1 # For complex questions that need larger context, choose 2
  selected_question = 0 # Choose the question you want to try by changing the inde

  question = questions[selected_question]

  # Find the game of the question
  results = collection.query(

  query_texts=[question],
  n_results=3,

  )

```

```
  # Get the name of the game (the most common value)
  games = [result['bg_name'] for result in results['metadatas'][0]]
  game_name = max(set(games), key=games.count)

  question = question.replace(game_name, "the game", -1)

  # Get results for the specific game

  results = collection.query(
  query_texts=[question],

  n_results=number_of_results,

  where={"bg_name": game_name},
  )

```

Here, you can play a little with `number_of_results` and `extend_results_by`, as stated in the comments of the code, in order to get better answers.

Since we only got the most relevant chunks from our semantic search, there is a high probability that we did not get all the relevant information we needed. As a solution, we select surrounding chunks (previous and next chunks for each chunk) to try and get the full context.

```
  # Select surrounding documents/chunks (previous and next for each document), in

  ids = results['ids'][0]

  # Extracting the numbers from the ids and converting to integers

  numbers = [int(id.split('_')[-1]) for id in ids]

  # Getting the surrounding ids (chunks) based on the offset

  offset = extend_results_by
  result_list = []

  for id in ids:

  num = int(id.split('_')[-1])

  filename = id.rsplit('_', 1)[0]
     for i in range(-offset, offset + 1):

  surrounding_id = f'{filename}_{num + i:02d}'
  result_list.append(surrounding_id)

  result_list = list(dict.fromkeys(result_list))

  # Get the extended results.

  # Unfortunately, these come back sorted by id, so our semantically relevant chun
  results = collection.get(ids=result_list)

```

Now we just merge the returned relevant chunks of text.

```
  # Create the final context

  context = ' '.join(results['documents'])

```

It's time to use our model! In order to do that we will need several libraries [from Hugging Face](https://huggingface.co/).


To install use

```
  pip install transformers
  pip install bitsandbytes

  pip install sentencepiece

  pip install accelerate

```

To import use

```
  from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

  import sentencepiece
  import accelerate

```

Now we need to create our model and tokenizer using a quantization config.

Getting into too much explanation about quantization, tokenizer and model is out of the scope of this article, but in a few words we have:

**Quantization config**: We will use double quantization with 4bit numbers that will be converted into 16 bits for the calculations.

**Tokenizer**: The part of the model that gets text, and based on a vocabulary converts it (encodes it) to tokens (numbers) that have specific embeddings (vectors of numbers). It also gets tokens and decodes them to actual text.

**Model**: This is the main part of the model. It gets the input tokens and replies with the output tokens.

```
  bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,

  bnb_4bit_quant_type='nf4',

  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_use_double_quant=True,

  )

  model_name='microsoft/Phi-3-mini-128k-instruct'

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  model = AutoModelForCausalLM.from_pretrained(model_name,
  device_map="auto",

  quantization_config=bnb_config,

  trust_remote_code=True,
  )

```

The `trust_remote_code=True` argument is something that Phi3 needs.

Now we need to construct our input text that we will give to the model.

```
  input_text = '''<|user|>
  Given the following context and question, answer the question based on the conte

  Follow this procedure step by step.
  1. From the given context, based on the question, find only the relevant sentenc

  2. Based on the previous step, remove all the non relevant context from the give
  3. Answer ONLY based on the remaining context and nothing else.

  Context: ''' + context + '''

  Question: ''' + question + '''

  Answer:

  <|assistant|>

  '''

```

This is a humble try in prompt engineering here. Prompt engineering is the way to construct a question in order to give specific instructions to the model, so we can get a correct answer. The `<|user|>` and `<|assistant|>` special tokens here are how Phi3 expects input to be formatted and it is all explained in the Hugging Face Phi3 models pages.

Now that we have our full input text with context and question, we need to tokenize it and give it to the model to generate an answer. Another word for generation is inference.

```
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')

  outputs = model.generate(input_ids, max_new_tokens=20000)

```

`max_new_tokens` is the limit of the tokens the model will reply with. Have in mind that the model replies with the full initial input together with the answer.


Let’s decode and see the answer.

```
  outputs_list = []
  outputs_list.append(tokenizer.decode(outputs[0]).replace("<|end|>", ""))
  outputs_list[-1].split("<|assistant|>")[-1]

```

Our answer is the text after the special token `<|assistant|>` .

## **_Results_**

Because we had to choose a very small model, we could not provide too much context to the LLM.

During tests where we provided too much content, the LLM got confused and produced serious hallucinations.

So, you can ask questions that can be answered in 300–600 words in the rulebook.

It's better to avoid more complex questions like "For game X, please tell me how to setup a 2-player game".

However, if you want to try more complex questions, play with the variables `number_of_results` and `extend_results_by`.

Have in mind that larger values need more VRAM and increase execution time exponentially.

## **_With RAG_**

These are the RAG answers. Even the setup one is kind of OK. There is some information missing, but the setup of this game is very complex and some setup information is formatted very badly with nested lists and columns in the pdf rulebook. With a human corrected markdown rulebook, and perhaps a slightly larger model, even this question can be answered correctly. The other two answers are 100% correct and very well explained.

## **_The Witcher: Old World_**


_For The Witcher: Old World, please tell me how to setup a 2-player game._


To setup a 2-player game for The Witcher: Old World, follow these steps:


1. Setup the Game Board in the middle of the table.


2. Create the Action Card Deck by shuffling all Action Cards and placing


them facedown on the indicated spot near the Game Board.


3. Form the Action Cards Pool by revealing Action Cards until 3 “0-cost”


cards are revealed. Place these 3 cards face up on the 3 right-most slots


on the Game Board.


4. Shuffle and place the remaining revealed Action Cards on the indicated


spot near the Game Board.


5. Place 3 Action Cards from the Action Deck on the three remaining slots


on the Game Board.


6. Setup the Potion Deck by shuffling all Potion Cards and placing them face


down on their indicated space near the Game Board.


7. Place 1 Attribute Trophy Card for each Attribute (Combat, Defense,


Alchemy, and Specialty) on their designated spot on the Game Board.


8. Place the remaining 4 cards back in the box.


9. Determine the Starting Player. The Player who most recently read a


Witcher book is the Starting Player; alternatively, you can determine this


randomly.


10. Each Player takes an Action and Fight Help Card.


11. The Starting Player chooses the Terrain type for the Level II Monster


12. Setup for a 2-Player Game is complete.


_In The Witcher: Old World, can I play poker and fight a player at the same turn?_


Yes, in the same turn, you can choose to play Dice Poker with another


Witcher and also decide to Fight with a Witcher. However, you cannot Fight


with a Witcher you played Dice Poker with in the same turn.

## **_Dune: Imperium — Uprising_**


_In Dune: Imperium — Uprising, please explain in detail how spies and_


_observation posts work._


In the game, spies and observation posts are strategic elements that allow


players to gather intelligence and potentially infiltrate other players’


territories. Here’s a detailed explanation:


1. Observation Posts: These are various board spaces across the game


board. Each observation post is connected to one or more board spaces.


Whenever the Spy icon appears on a card or board space, a player may


place a Spy from their supply on an unoccupied observation post on the


board.


2. Placement of Spies: A player can only place a Spy on an observation post


if they have a Spy in their supply. If a player has no Spies in their supply


when they need to place one, they may first recall one of their Spies for


no effect.


3. Recall of Spies: Spies can be recalled from an observation post without


using a card. There are two effects of recalling a Spy:


Infiltrate: If a player wishes to send an Agent to a board space occupied


by another player, they may recall their own Spy from a connected


observation post to ignore the other player’s Agent and send their Agent


to that same board space.


Gather Intelligence: Whenever a player sends an Agent to a board space,


they may recall their own Spy from a connected observation post to draw


a card. This can only be done immediately after placing the Agent.


4. Spy Agent Icon: This allows a player to send an Agent to a board space


connected to an observation post where they currently have a Spy. However,


the Spy is not recalled for this purpose.


In summary, spies and observation posts add a layer of strategy to the game,


allowing players to gather intelligence and potentially disrupt other players’


plans. They are an integral part of the gameplay and can significantly


influence the outcome of the game.

## **_Without RAG_**

Here we can see that the model went into full hallucination mode, as it doesn't have any information on these games. This is natural, as this is a very small model, trained only in the most important datasets. For the Witcher setup question, I find very funny the "Purchase the Game" step. In the Dune Imperium — Uprising, it even managed to spell the game name incorrectly. I understand that you need to know the games and the rules to see how wrong these answers are, but take my word, they are very wrong. As you can see though, they are extremely believable. This is the definition of an LLM hallucination.

## **_The Witcher: Old World_**


_For The Witcher: Old World, please tell me how to setup a 2-player game._


To set up a 2-player game for “The Witcher: Old World”, follow these steps:


1. Purchase the Game: Ensure you have two copies of “The Witcher: Old


World” for each player.


2. Install the Game: Install the game on your respective systems. If you’re


playing on different platforms (PC, PlayStation, Xbox, etc.), you’ll need to


install the game on each of your systems.


3. Connect Online: If the game supports online play, you’ll need to connect


to the same online gaming service (like PlayStation Network or Xbox


Live) to play together.


4. Create Accounts: Create separate accounts for each player if required by


the game.


5. Start a Game: Launch the game on both systems. You can either start a


new game or load an existing one.


6. Join the Game: On the system where you’re hosting the game, you’ll need


to join the game session. This is usually done through the game’s main


menu or a specific ‘Join Game’ option.


7. Play Together: Once joined, you can play together, with each player


controlling their own Geralt of Rivia character.


Remember to check the game’s specific instructions or community forums


for any additional steps or tips specific to “The Witcher: Old World”.


_In The Witcher: Old World, can I play poker and fight a player at the same turn?_


No, in The Witcher: Old World board game, you cannot perform two actions


that require different types of resources in the same turn. Poker is a card



[game that requires gold, while fighting a player is a combat action that](https://rsci.app.link/?%24canonical_url=https%3A%2F%2Fmedium.com%2Fp%2Fa0a1de586209&%7Efeature=LiOpenInAppButton&%7Echannel=ShowPostUnderUser&%7Estage=mobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------) _[Open in app](https://rsci.app.link/?%24canonical_url=https%3A%2F%2Fmedium.com%2Fp%2Fa0a1de586209&%7Efeature=LiOpenInAppButton&%7Echannel=ShowPostUnderUser&%7Estage=mobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------)_


requires a sword.

_Search_ _[Write](https://medium.com/new-story?source=post_page---top_nav_layout_nav-----------------------------------------)_

## **_Dune: Imperium — Uprising_**


_In Dune: Imperium — Uprising, please explain in detail how spies and_


_observation posts work._



_10_



In the “Dune: Imerium — Uprising” board game, spies play a crucial role in


gathering information and gaining strategic advantages. Here’s a detailed


guide on how you can use spies effectively:


1. Understanding Spy Mechanics: Spies are represented by the Spy tokens in


the game. Each player starts with a certain number of these tokens,


depending on their faction. Spies can be used to gather information about


opponents’ plans, resources, and positions.


2. Gathering Information: To use a spy, you need to place it on a specific type


of space called an Information Space. These spaces are usually marked with


a symbol or color indicating their purpose. Information Spaces are found on


the game board and are strategically placed to provide information about


different aspects of the game.


3. Information Gathering: When you place a spy on an Information Space, it


will reveal information about the opponent’s resources, plans, or positions.


The information gained can be used to make informed decisions about your


own strategy.


4. Information Usage: The information gathered by spies can be used in


various ways. For example, you can use it to plan your moves, predict


opponents’ actions, or even sabotage their plans. Information can also be


used to gain advantages in resource allocation or to influence the outcome of


certain events in the game.


5. Limitations and Risks: Using spies comes with certain limitations and


risks. Spies can only be used once per turn, and they cannot be used to


directly attack opponents. Additionally, spies can be detected by opponents,


which may lead to countermeasures or retaliation.


6. Strategic Placement: To maximize the effectiveness of spies, it’s important


to strategically place them on Information Spaces. Consider the information


you need to gather and the opponents you want to monitor. Placing spies in


locations that provide valuable information about multiple opponents can be


particularly advantageous.


7. Balancing Spy Usage: While spies can provide significant advantages, it’s


important to balance their usage with other gameplay elements. Over

reliance on spies may lead to predictability in your strategy, making it easier


for opponents to counter.


In summary, spies in “Dune: Imerium — Uprising” are a powerful tool for


gathering information and gaining strategic advantages. By understanding


their mechanics, strategically placing them, and using the information


effectively, you can enhance your gameplay and increase your chances of


success.

## **_Conclusion_**

We managed to create a full open source simple RAG system, that includes everything. Correctly parse PDF files, split the text into chunks, save the chunks with metadata as embeddings in a local vector store, retrieve relevant to the question chunks from the vector store, extend the chunks in a try to have all the relevant information, and finally get a correct answer. All that done completely for free, on a local low VRAM environment, using only Hugging Face libraries (and a little Llama Index for the parsing) and keeping full control in the process.

I hope you enjoyed this article.

Happy coding!


_[Rag](https://medium.com/tag/rag?source=post_page-----a0a1de586209---------------------------------------)_ _[Open Source](https://medium.com/tag/open-source?source=post_page-----a0a1de586209---------------------------------------)_ _[Llm](https://medium.com/tag/llm?source=post_page-----a0a1de586209---------------------------------------)_ _[Python](https://medium.com/tag/python?source=post_page-----a0a1de586209---------------------------------------)_ _[Hugging Face](https://medium.com/tag/hugging-face?source=post_page-----a0a1de586209---------------------------------------)_


_Written by Elias Tarnaras_


_[3 followers · 1 following](https://medium.com/@elias.tarnaras/followers?source=post_page---post_author_info--a0a1de586209---------------------------------------)_

## _No responses yet_


_Sébastien Morand_


_What are your thoughts?_

## _More from Elias Tarnaras_


_[Elias Tarnaras](https://medium.com/@elias.tarnaras?source=post_page---author_recirc--a0a1de586209----0-----------------------------------)_

## **_YouTube History Semantic Search_**


_Using embeddings, re-ranking, ChromaDB, FastAPI, Gradio_



_Follow_



_Jun 5, 2024_



_4_



_[See all from Elias Tarnaras](https://medium.com/@elias.tarnaras?source=post_page---author_recirc--a0a1de586209---------------------------------------)_


## _Recommended from Medium_

_[Md. Tariqul Islam](https://medium.com/@tisbuet_18039?source=post_page---read_next_recirc--a0a1de586209----0---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_


_**[LightRAG is lite RAG](https://medium.com/@tisbuet_18039/lightrag-is-lite-rag-3468ef91cba7?source=post_page---read_next_recirc--a0a1de586209----0---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)**_


_[Since the emergence of OpenAI’s LLM](https://medium.com/@tisbuet_18039/lightrag-is-lite-rag-3468ef91cba7?source=post_page---read_next_recirc--a0a1de586209----0---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_
_models, Retrieval Augmented…_



_[Faris M. Syariati](https://medium.com/@farissyariati?source=post_page---read_next_recirc--a0a1de586209----1---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_


_**Ask Your Codebase Anything Using**_
_**Ollama, Embeddings, and RAG**_


_[In this tutorial, we are going to create a simple](https://medium.com/@farissyariati/ask-your-codebase-anything-using-ollama-embeddings-and-rag-c65081a5ef20?source=post_page---read_next_recirc--a0a1de586209----1---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_
_Retrieval-Augmented Generation (RAG)…_



_Feb 22_



_8_ _1_



_May 4_



_8_ _1_



_[Andreas Klos](https://medium.com/@a-klos?source=post_page---read_next_recirc--a0a1de586209----0---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_


_**A Simple Guide to OCR with Vision**_
_**LLMs, LangChain, and Ollama**_



_[S. Hassan Tabatabaei](https://medium.com/@hassan.tbt1989?source=post_page---read_next_recirc--a0a1de586209----1---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_


_**Build a RAG-Powered LLM Service**_

_**[with Ollama & Open WebUI : A…](https://medium.com/@hassan.tbt1989/build-a-rag-powered-llm-service-with-ollama-open-webui-a-step-by-step-guide-a688ec58ac97?source=post_page---read_next_recirc--a0a1de586209----1---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)**_

_Introduction_ 🚀



_Apr 9_



_103_ _1_



_Feb 13_



_35_ _2_


_In_ _[AI Advances](https://medium.com/ai-advances?source=post_page---read_next_recirc--a0a1de586209----2---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_ _by_ _[Anjolaoluwa Ajayi](https://medium.com/@dataprincess?source=post_page---read_next_recirc--a0a1de586209----2---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_



_**[21 Chunking Strategies for RAG](https://medium.com/ai-advances/21-chunking-strategies-for-rag-f28e4382d399?source=post_page---read_next_recirc--a0a1de586209----2---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)**_


_[And how to choose the right one for your next](https://medium.com/ai-advances/21-chunking-strategies-for-rag-f28e4382d399?source=post_page---read_next_recirc--a0a1de586209----2---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_
_LLM application_



_In_ _[Product & Engineering at S…](https://medium.com/showpad-engineering?source=post_page---read_next_recirc--a0a1de586209----3---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_ _by_ _[Pieterjan Criel …](https://medium.com/@pjcr?source=post_page---read_next_recirc--a0a1de586209----3---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_


_**Contextual embedding for RAG**_
_**with Fence**_ 🤺 _**Ollama**_ 🦙 _**and…**_

✨
_[Foundational models rely on a lot of data for](https://medium.com/showpad-engineering/contextual-embedding-for-rag-with-fence-ollama-and-chromadb-967330a0080e?source=post_page---read_next_recirc--a0a1de586209----3---------------------d6e49a9c_a8cc_45d4_a392_fada0a2d02d8--------------)_
_training. Andrej Karpathy published an…_



_Jun 30_



_1.2K_ _12_



_Feb 23_



_78_



_[See more recommendations](https://medium.com/?source=post_page---read_next_recirc--a0a1de586209---------------------------------------)_




import langchain
from youtube_transcript_api import YouTubeTranscriptApi
import tiktoken
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from dataclasses import dataclass
from typing import List, Dict
from langchain.llms import OpenAIChat
import os
import heapq
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)

enc = tiktoken.get_encoding("gpt2")
MAX_NUM_TOKEN_PER_PIECE = 500

chat = ChatAnthropic(model="claude-v1")
chat_100k = ChatAnthropic(model="claude-v1-100k")


@dataclass
class Slice:
  num_tokens: int
  text: str
  transcripts: List[Dict]
  requirements_str: str
  index: int

  def __lt__(self, other):
    ret = compare_pieces(self.requirements_str, self, other)
    # assuming compare is defined elsewhere to handle Slice objects
    print(
      f"[Compare] between {self.index} and {other.index} gets result {ret}")
    return ret


def get_video_transcript(video_id):
  transcripts = YouTubeTranscriptApi.get_transcript(video_id)
  return transcripts


def slice_transcript(total_transcripts, requirements_str):
  total_slices = []
  current_num_tokens = 0
  current_transcripts = []
  current_text = ""
  for transcript in total_transcripts:
    text = transcript["text"]
    num_tokens = len(enc.encode(text))
    if num_tokens + current_num_tokens > MAX_NUM_TOKEN_PER_PIECE:
      # create a new piece
      slice = Slice(num_tokens=current_num_tokens,
                    text=current_text,
                    transcripts=current_transcripts,
                    requirements_str=requirements_str,
                    index=len(total_slices))
      total_slices.append(slice)
      current_num_tokens = 0
      current_transcripts = []
      current_text = ""
    else:
      current_num_tokens = current_num_tokens + num_tokens
      current_text = current_text + "\n" + text
      current_transcripts.append(transcript)
  if len(current_transcripts) > 0:
    slice = Slice(num_tokens=current_num_tokens,
                  text=current_text,
                  transcripts=current_transcripts,
                  requirements_str=requirements_str,
                  index=len(total_slices))
    total_slices.append(slice)
    current_num_tokens = 0
    current_transcripts = []
    current_text = ""
  return total_slices


# Given two pieces and a set of requirements, find the piece that's more aligned with the requirement
# Return 0 or 1 or 2, 0 meaning they are equally meet the requirement, 1 means piece_a is more relevant, 2 means
def compare_pieces(requirements_prompt, piece_a, piece_b):
  text_a = piece_a.text
  text_b = piece_b.text

  prompt = f"""
    You are a helpful assistant to compare two pieces of content and identify the better one according to the set of requirements.
    Please always only return 0 or 1 or 2 as the answer, return 0 if there is no answer found, no additional word is needed in the answer

    Below are some example: 

    Requirements:
    1. Find the piece of content that mention 'the more you buy, the more you save' 

    Content 1: 
    I always say the more you buy the more you save but it's not always the case thus 
    I never say that in the rest of my life

    Content 2:
    You should always follow the AI trend, since it will give you very big benefit in your life

    Answer:
    1

    Requirements:
    1. Find the piece of content that's emotionally sad 

    Content 1: 
    I always tell my friend that I have a good day and they should be grateful about their life
    

    Content 2:
    Life is a box of candies full of joy and surprises

    Answer:
    0


    Requirements:
    {requirements_prompt}

    Content 1:
    {text_a}

    Content 2:
    {text_b}

    Answer:"""
  messages = [HumanMessage(content=prompt)]
  result = chat(messages)
  answer = result.content
  if int(answer) == 0:
    return 0
  elif int(answer) == 1:
    return 1
  elif int(answer) == 2:
    # Meaning the other one is bigger
    return -1
  else:
    print(f"Got unexpected answer {answer}")
    return 0


def traverse_slices(requirements, slices):
  if len(slices) == 0:
    return None
  elif len(slices) == 1:
    return slices[0]
  elif len(slices) == 2:
    return compare_pieces(requirements_prompt=requirements,
                          piece_a=slices[0],
                          piece_b=slices[1])
  else:
    # Split into two chunks and conqure
    # calculate the index that splits the list in half
    mid_index = len(slices) // 2

    # split the list
    first_half = slices[:mid_index]
    second_half = slices[mid_index:]
    result = []
    reduced_left = traverse_slices(requirements=requirements,
                                   slices=first_half)
    if reduced_left is not None:
      result.append(reduced_left)
    reduced_right = traverse_slices(requirements=requirements,
                                    slices=second_half)
    if reduced_right is not None:
      result.append(reduced_right)
    if len(result) == 0:
      return None
    elif len(result) == 1:
      return result[0]
    else:
      return compare_pieces(requirements_prompt=requirements,
                            piece_a=result[0],
                            piece_b=result[1])


# Given the url and a list of requirments, sort and find the top k elements that meet the requirement
def get_top_k_relevant(url, top_k, requirements_str):
  video_id = _extract_video_id(url)
  print(f"Extracted video id {video_id} from url {url}")
  transcripts = get_video_transcript(video_id)
  print(f"Successfully fetched transcripts for video {video_id}, got {len(transcripts)} transcripts")
  
  slices = slice_transcript(total_transcripts=transcripts,
                            requirements_str=requirements_str)[:10]
  heap = []
  for slice in slices:
    if "more you save" in slice.text:
      print(f"save exist in slice {slice.index}")

    if len(heap) == top_k:
      # This seems to be a bug of heap, if there is only the top element is bigger than all other elements in the heap, and all other elements are equal, then the new element will cause the top element to be pushed out
      if slice > heap[-1]:
        popped = heapq.heapreplace(heap, slice)
        print(f"[Pop]  {popped.index} got popped as the smallest")
      else:
        print(
          f"No pop is needed, since {slice.index} is smaller than {heap[-1].index}"
        )
    else:
      heapq.heappush(heap, slice)
    print(f"[Layout] heap is {[slice.index for slice in heap]}")
  return [slice for slice in heap]


def _extract_video_id(url):
    # Check if the URL is a valid YouTube URL
    if "youtube.com/watch?v=" not in url:
        return None
    
    # Find the start index of the video ID
    start_index = url.index("?v=") + 3
    
    # Find the end index of the video ID (could be '&' or end of string)
    end_index = url.index("&") if "&" in url else len(url)
    
    # Extract the video ID using the start and end indices
    video_id = url[start_index:end_index]
    
    return video_id

def ask_video_question(url, question):
  video_id = _extract_video_id(url)
  print(f"Extracted video id {video_id} from url {url}")
  transcripts = get_video_transcript(video_id)
  print(f"Successfully fetched transcripts for video {video_id}, got {len(transcripts)} transcripts")
  context = "\n".join([transcript["text"] for transcript in transcripts])

  prompt = f""" 
    You are a helpful assistant helping to answer questions based on the context, the context a full transcript of a video. 

    Questions: 
    {question}


    Context: 
    {context}


    """

  messages = [HumanMessage(content=prompt)]
  result = chat_100k(messages)
  return result.content


#### Example 1: 

print(ask_video_question(url="https://www.youtube.com/watch?v=pdJQ8iVTwj8", question="Please summarize the video into 5 bullet points"))


#### Example 2:

# """
# Useful references:
# * Original video: https://www.youtube.com/watch?v=i-wpzS9ZsCs
# * Good clip: https://www.youtube.com/watch?v=_SloSMr-gFI 
# * 

# """
# ### Test case 1:
# requirements_str = """
# 1. Find the piece of content that mention 'the more you buy, the more you save' 
# """

# results = get_top_k_relevant(url="https://www.youtube.com/watch?v=pdJQ8iVTwj8",
#                              top_k=3,
#                              requirements_str=requirements_str)

# top_result = results[0]

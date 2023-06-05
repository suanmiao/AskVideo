## Anthropic Hackathon Project: Ask Video

Hey, welcome to check out the small project I built during the 2023 Anthropic hackathon.

Unfortunately I don't have enough time to build a web UI for this app, but the idea is pretty cool!
- Given a youtube video, you can ask any question about it!
  - You can have a try through the function ask_video_question
- Given a youtube video, you can semantically find the most relevant pieces according to a set of rules. This has very huge potential!
  - You can easily find all pieces that's related to certain topic, basically use it like a copilot for video
  - With good enough rules, you can even let it find the most attractive pieces, then it can be used as a video compresser (by generating a shorter video from the long video)


This is really a chance for me to get familiar with Anthropic API, langchain and youtube transcript. It's so great to try out the claude-100k model, which is like magic!


### Get started

Simply modify the main.py at the end, and then run it.
- I have the functions all implemented in that file already
- And I also have provided correspoinding examples for ask_video_question and get_top_k_relevant



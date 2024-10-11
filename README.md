# Blog Generation System

The **Blog Generation System** is an automated tool that generates blog content using AI-driven language models like GPT. This system integrates with external sources such as Wikipedia and Google Search to gather the latest information and enhance the quality of the generated blogs.

## Features
- **Real-time Data Integration:** Pulls information from Google Search and Wikipedia to create fact-based blog content.
- **Customizable Prompts:** Supports dynamic user prompts to generate content for specific niches.
- **Human-like Text Generation:** Utilizes GPT-based models to produce coherent, readable, and engaging blog posts.
- **Scalable and Extensible:** Easily extendable with more data sources or customization of text generation rules.

## Prerequisites

Ensure you have the following installed before starting:
- Python 3.10+
- A virtual environment setup (recommended)
- API access keys for:
  - OpenAI GPT models
  - Google Search API
  - Wikipedia API

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/blog-generation-system.git
   cd blog-generation-system
2.Set up a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

3.Install the dependencies:
pip install -r requirements.

4.Add your API Keys:
Google API Key: Add your key to a .env file or in the settings section of the code.
OpenAI API Key: Similarly, add your GPT API key in the .env file.

5.Usage
Run the script:
streamlit run your_script.py
Customize the inputs: You can customize the prompt inputs for the model to generate content for various blog topics.

Enjoy your generated blog content: The system will pull relevant information and use GPT to generate the final blog post.

We welcome contributions! Please fork the repository and create a pull request with your changes.

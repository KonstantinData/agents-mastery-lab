# M4 Agentic AI - Evaluation Methods in Research Reports

## 1. Introduction

### 1.1. Lab overview

In this lab, you will build on the workflow introduced in the graded lab at the end of **Module 3**.  
The focus here is on how to integrate and apply **evaluation methods** inside a workflow that generates a short research report.  

### 1.2. 🎯 Learning outcomes

You will learn how to:

* Write a function that can check the search results of a web search API for **preferred sources**.  
* Create an evaluation to verify if your sources come from your **preferred domains**.  
* Add a **component-level evaluation** to the `find_references` function.  


## 2. Setup: Import libraries and load environment

As in previous labs, you start by importing the required libraries and initializing your environment.


```python
# =========================
# Imports
# =========================

# --- Standard library 
from datetime import datetime
import json
import re

# --- Third-party ---
from aisuite import Client

# --- Local / project ---
import research_tools
import utils

client = Client()
```

## 3. Steps — Building your toolkit

In this section, you will define the **steps** of the workflow. Each step plays a specific role:

* **Step 1: find_references** — gathers information  
* **Step 2: write_draft** — drafts the report  
* **Step 3: reflect_and_rewrite** — improves the draft  

By combining these steps, you can build a workflow that turns a topic into a polished research report.

### 3.1. Step 1: find_references

As part of the workflow, you will now introduce the **research step**.  Its role is to **gather external information** using tools such as Arxiv, Tavily, and Wikipedia.  
this function will search various websites for articles and other resources that are relevant to your query.


```python
def find_references(task: str, model: str = "openai:gpt-4o", return_messages: bool = False):
    """Perform a research task using external tools (arxiv, tavily, wikipedia)."""

    prompt = f"""
    You are a research function with access to:
    - arxiv_tool: academic papers
    - tavily_tool: general web search (return JSON when asked)
    - wikipedia_tool: encyclopedic summaries

    Task:
    {task}

    Today is {datetime.now().strftime('%Y-%m-%d')}.
    """.strip()

    messages = [{"role": "user", "content": prompt}]
    tools = [
        research_tools.arxiv_search_tool,
        research_tools.tavily_search_tool,
        research_tools.wikipedia_search_tool,
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_turns=5,
        )
        content = response.choices[0].message.content
        return (content, messages) if return_messages else content
    except Exception as e:
        return f"[Model Error: {e}]"
```

Run the following cell to try out the **research function**.  You will ask it to find two recent papers about neural networks on arXiv and then display the results.


```python
research_task = "Find 2 recent papers about recent developments in black hole science"
research_result = find_references(research_task)

utils.print_html(
    research_result[:300] + "..." if len(research_result) > 300 else research_result,
    title="Research Function Output"
)
```

### 3.2. Step 2: write_draft

This step generates a structured academic or technical draft based on your research results.  
Its role is to create a clear, organized first version of the report.  

You will use this step whenever you need to transform your collected references into a coherent draft report.  



```python
def write_draft(task: str, model: str = "openai:o4-mini") -> str:
    """Generate a well-structured academic/technical draft based on the given task."""

    messages = [
        {
            "role": "system",
            "content": "You are a writing function specialized in clear and well-structured academic/technical content."
        },
        {"role": "user", "content": task},
    ]
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0
    )
    return resp.choices[0].message.content

```

Run the following cell to try out the **writer function**.  You will ask it to create a short technical report about quantum computing, organized into four sections: Introduction, Key Principles, Applications, and Conclusion.


```python
writing_task = "Write a brief technical report about recent developments in black hole science: Introduction, Key Principles, Applications, Conclusion"

draft_result = write_draft(writing_task)

utils.print_html(
    draft_result[:400] + "..." if len(draft_result) > 400 else draft_result,
    title="Writer Function Output"
)
```

### 3.3. Step 3: reflect_and_rewrite

This step reviews the draft created by the `write_draft` step.  
It evaluates the text for clarity, structure, and coherence, then applies improvements to produce a clearer and higher-quality report.  

You will use this step whenever you need to refine a draft into its final polished version.  



```python
def reflect_and_rewrite(task: str, model: str = "openai:o4-mini") -> str:
    """Reflect on, critique, and improve a draft to produce a clearer, higher-quality version."""

    messages = [
        {
            "role": "system",
            "content": "You are a reflection and rewrite function. Reflect on, critique, and improve drafts."
        },
        {"role": "user", "content": task},
    ]
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1
    )
    return resp.choices[0].message.content
```

Run the following cell to try out the **reflection and rewrite function**.  You will provide it with a simple draft and ask it to improve clarity, structure, and quality.


```python
draft_to_edit = (
    "Black holes are very mysterious. They are heavy and strong. "
    "Scientists study them because they are interesting. "
    "Sometimes they are in movies. We don’t know everything about them."
)

editing_task = f"Improve this draft: {draft_to_edit}"
edited_result = reflect_and_rewrite(editing_task)

utils.print_html(
    edited_result[:300] + "..." if len(edited_result) > 300 else edited_result,
    title="Reflection & Rewrite Output"
)
```

## 4. Evaluate the web domains returned by the research step

In the video, Andrew explored the case where web search results were of poor quality.  
Now, you’ll create a **component-level evaluation** that counts how many of the web sources returned by the `find_references` step belong to your list of **preferred domains**.  

For this evaluation case, you will focus on the topic *“recent developments in black hole science”*, one of the examples shown in the course.  

This evaluation will take the form of a single function that performs an **objective check** with a **per-example ground truth**.  
Specifically, it will:

- Parse the Tavily output (JSON string or list of dicts).  
- Check how many URLs belong to a predefined allow-list of **preferred domains** (`TOP_DOMAINS`).  
- Compute the ratio of preferred vs. total results.  
- Return a boolean flag (**PASS/FAIL**) together with a detailed Markdown summary that can be directly included in reports.  

<img src='M4-UGL-1.png' width=50%></img>

<div style="border:1px solid #fca5a5; border-left:6px solid #ef4444; background:#fee2e2; border-radius:6px; padding:12px 14px; color:#7f1d1d; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
  <strong>🔎 Why this is an objective evaluation:</strong><br><br>
  Each URL retrieved from Tavily is compared against a predefined allow-list of <em>preferred domains</em> (<code>TOP_DOMAINS</code>):<br>
  • If the domain matches → ✅ PREFERRED<br>
  • Otherwise → ❌ NOT PREFERRED<br><br>
  This yields a clear PASS/FAIL signal based on whether the ratio of preferred sources exceeds the threshold.  
  Because the ground truth (preferred vs. not preferred) is explicitly defined per example, the evaluation is objective and reproducible.
</div>

In the next cell, you will find the definition of this function.  



```python
TOP_DOMAINS = {
    # General reference / institutions / publishers
    "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
    "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

    # CS/AI venues & indexes
    "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

    # Other reputable outlets
    "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

    # Extra domains (case-specific additions)
    "pbs.org", "nova.edu", "nvcc.edu", "cccco.edu",

    # Well known programming sites
    "codecademy.com", "datacamp.com"
}

def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio=0.4):
    """
    Evaluate whether plain-text research results mostly come from preferred domains.

    Args:
        TOP_DOMAINS (set[str]): Set of preferred domains (e.g., 'arxiv.org', 'nature.com').
        raw (str): Plain text or Markdown containing URLs.
        min_ratio (float): Minimum preferred ratio required to pass (e.g., 0.4 = 40%).

    Returns:
        tuple[bool, str]: (flag, markdown_report)
            flag -> True if PASS, False if FAIL
            markdown_report -> Markdown-formatted summary of the evaluation
    """

    # Extract URLs from the text
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, """### Evaluation — Tavily Preferred Domains
No URLs detected in the provided text. 
Please include links in your research results.
"""

    # Count preferred vs total
    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        domain = url.split("/")[2]
        preferred = any(td in domain for td in TOP_DOMAINS)
        if preferred:
            preferred_count += 1
        details.append(f"- {url} → {'✅ PREFERRED' if preferred else '❌ NOT PREFERRED'}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio

    # Markdown report
    report = f"""
### Evaluation — Tavily Preferred Domains
- Total results: {total}
- Preferred results: {preferred_count}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}

**Details:**
{chr(10).join(details)}
"""
    return flag, report

```

<div style="border:1px solid #93c5fd; border-left:6px solid #3b82f6; background:#dbeafe; border-radius:6px; padding:12px 14px; color:#1e3a8a; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
Run the next cell to see a small sample of preferred domains and evaluate two URLs (one preferred, one not).
</div>




```python
utils.print_html(json.dumps(list(TOP_DOMAINS)[:4], indent=2), title="Sample Trusted Domains")

utils.print_html("<h3>Research Results</h3>" + research_result, title="Research Results")

flag, report = evaluate_tavily_results(TOP_DOMAINS, research_result)
utils.print_html("<pre>" + report + "</pre>", title="<h3>Evaluation Summary</h3>")

```

## 5. End-to-End: Run the Workflow

In this final step, you will run the **full workflow** to generate a short research report. The process will:

a) Use the **research function** to gather information from external sources (Tavily, Wikipedia, arXiv).  

b) Run an **evaluation step** to check whether your sources come from preferred domains.  

c) Pass the results to the **writer function** to create a first draft in Markdown.  

d) (Optional) Improve the draft with the **reflection and rewrite function** for clarity and style.  

By default, the evaluation uses a threshold of `min_ratio = 0.4` (40%). This means at least 40% of Tavily results must come from preferred `TOP_DOMAINS`. You can adjust this ratio if you want the evaluation to be stricter or more relaxed.

You will now try it out with the topic **Ensemble Kalman filter** and follow the workflow end to end.





```python
# 1) Run research
topic = "recent developments in black hole science"
research_task = f"Find 2–3 key papers and trusted overviews about {topic}."
research_output = find_references(research_task)

utils.print_html(research_output, title=f"<h3>Research Results on {topic}</h3>")

# 2) Evaluate sources (Tavily references only)
flag, eval_md = evaluate_tavily_results(TOP_DOMAINS, research_output, min_ratio=0.4)
utils.print_html("<pre>" + eval_md + "</pre>", title="<h3>Evaluation Summary</h3>")

# 3) Draft the report
writing_task = (
    f"Write a concise Markdown research report about {topic} using ONLY these research results:\n{research_output}"
)
draft = write_draft(writing_task)
utils.print_html("<pre>" + draft + "</pre>", title="<h3>Draft Report</h3>")

# 4) Optional reflection & rewrite
final_report = reflect_and_rewrite(f"Improve this draft for clarity and structure (return Markdown only):\n{draft}")
utils.print_html("<pre>" + final_report + "</pre>", title="<h3>🧾 Reflection & Rewrite Report</h3>")


```

## 6. Takeaways

* **Evaluation is essential**: explicitly checking your sources against preferred domains ensures that your research is grounded in reliable references.  
* **Confidence in results**: evaluation steps provide a clear PASS/FAIL signal, helping you decide whether to trust the output or refine your queries.  
* **Graceful degradation**: even if the evaluation fails, the workflow continues—showing that imperfect inputs can still be useful, but should be treated with caution.  
* **Better reports**: combining research steps with explicit evaluation and refinement leads to concise, clear, and credible research reports.  


<div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">

🎉 <strong>Congratulations!</strong>

You just built a research workflow that doesn’t stop at gathering information—it also <em>evaluates the quality of your sources</em>.  
By checking results against preferred domains, you introduced a simple but powerful safeguard that makes your reports more credible.  

You learned that component-level evaluation adds transparency and confidence: a clear PASS/FAIL signal tells you when to trust the output and when to refine your queries.  
Even when evaluation fails, the workflow continues, showing that you can still produce results while keeping quality in mind.  

With this skill, you’re ready to design <strong>workflows</strong> that not only generate content but also <strong>assess source reliability</strong>, ensuring your outputs are both useful and trustworthy. 🌟
</div>


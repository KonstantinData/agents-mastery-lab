# M2 - Agentic AI - SQL Generation

## 1. Introduction

### 1.1. Lab overview

In this lab, you will explore how **reflection patterns** can make agents reason more effectively and produce better SQL queries. You’ll see how an agent can spot issues in its own outputs, refine them, and improve its response before giving a final answer.

### 🎯 1.1 Learning outcome

You will practice applying reflection patterns to strengthen agent reasoning and outputs.

To do this, you will build **evaluation loops** where the agent:

* Reviews its own intermediate results (such as draft SQL or tool outputs).
* Identifies errors or gaps.
* Checks its responses and tool use.
* Refines the output before submitting the final answer.

## 2. Setup: Initialize Environment and Client

In this step, you will prepare your workspace so you can start coding right away. You will:

1. **Install the required Python libraries**
   Make sure all the dependencies are available so the code runs smoothly.

2. **Load environment variables from the `.env` file**
   These variables configure your environment (e.g., API keys or database settings).

3. **Import the `utils.py` module**
   This file contains helper functions that you will use to:

   * Create a sample products database.
   * Retrieve the database schema.
   * Run SQL queries.

**Note:** If you want to explore the contents of `utils.py`, go to the top menu and select **File > Open**.


```python
import json
import utils
import pandas as pd
import aisuite as ai
from dotenv import load_dotenv

_ = load_dotenv()
```

### 2.1 Getting started with AISuite
Now, initialize the `aisuite client`. This client gives you a single, unified way to connect and interact with different LLMs — so you don’t have to worry about each model having its own setup.


```python
client = ai.Client()
```

### 2.2 Set Up the Database

In this step, you will create a local SQLite database called **`products.db`**.
The database will be automatically filled with randomly generated product data.

You will use this data later in the lab to practice and test your SQL queries.


```python
utils.create_transactions_db()
```

You can inspect the table schema by executing the cell below.


```python
utils.print_html(utils.get_schema('products.db'))
```

## 3. Build a SQL generator

### 3.1 Use an LLM to Query a Database

In this step, you will use a function that turns your natural-language questions into SQL queries.

You provide your question and the database schema as input. The LLM then generates the SQL query that answers your question.

This way, **you** can focus on asking questions while the model takes care of writing the query.



```python
def generate_sql(question: str, schema: str, model: str) -> str:
    prompt = f"""
    You are a SQL assistant. Given the schema and the user's question, write a SQL query for SQLite.

    Schema:
    {schema}

    User question:
    {question}

    Respond with the SQL only.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

### 3.2 Reflect with External Feedback

Now, you will add a layer of quality control to **your** SQL queries.
When you run this step, you will get:

* A short evaluation of your query.
* Concrete suggestions for improvement.
* A refined SQL statement that better matches your question.


```python
def evaluate_and_refine_sql(
    question: str,
    sql_query: str,
    df: pd.DataFrame,
    schema: str,
    model: str,
) -> tuple[str, str]:
    """
    Evaluate whether the SQL result answers the user's question and,
    if necessary, propose a refined version of the query.
    Returns (feedback, refined_sql).
    """
    prompt = f"""
    You are a SQL reviewer and refiner.

    User asked:
    {question}

    Original SQL:
    {sql_query}

    SQL Output:
    {df.to_markdown(index=False)}

    Table Schema:
    {schema}

    Step 1: Briefly evaluate if the SQL output answers the user's question.
    Step 2: If the SQL could be improved, provide a refined SQL query.
    If the original SQL is already correct, return it unchanged.

    Return a strict JSON object with two fields:
    - "feedback": brief evaluation and suggestions
    - "refined_sql": the final SQL to run
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    
    content = response.choices[0].message.content
    try:
        obj = json.loads(content)
        feedback = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback if the model does not return valid JSON:
        # use the raw content as feedback and keep the original SQL
        feedback = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql
```

### 3.3 Automate a Comprehensive SQL Workflow

In this step, **you** will use a function that automates the entire workflow of creating, running, and improving SQL queries with an LLM.

The workflow walks you through key steps:

* Extracting the database schema.
* Generating a SQL query from **your** question.
* Executing the query.
* Evaluating its effectiveness.

If needed, the function will refine the query to better match **your** intent and run it again to ensure accuracy.

At the end, **you** will see:

* Both the initial and refined queries.
* Their results.
* Feedback from the LLM.

This makes it easier for **you** to work with SQL queries in a smoother, more accurate, and fully automated way.



```python
def run_sql_workflow(
    db_path: str,
    question: str,
    model_generation: str = "openai:gpt-4.1",
    model_evaluation: str = "openai:gpt-4.1",
):
    """
    End-to-end workflow to generate, execute, evaluate, and refine SQL queries.
    
    Steps:
    1. Extract the database schema.
    2. Generate a candidate SQL query from the user question.
    3. Execute the SQL query and show the initial result.
    4. Evaluate and refine the SQL query (if needed).
    5. Execute the refined query.
    """
    schema = utils.get_schema(db_path)
    utils.print_html("📘 Get schema:\n" + schema)

    sql = generate_sql(question, schema, model_generation)
    utils.print_html("🧠 Generate SQL (V1):\n" + sql)

    df = utils.execute_sql(sql, db_path)
    utils.print_html("📊 RExecute V1 query → Output:\n" + df.to_html())

    feedback, refined_sql = evaluate_and_refine_sql(
        question=question,
        sql_query=sql,
        df=df,
        schema=schema,
        model=model_evaluation,
    )
    utils.print_html("📝 Reflect on V1 SQL/output:\n" + feedback)
    utils.print_html("🔁 Write V2 query:\n" + refined_sql)

    refined_df = utils.execute_sql(refined_sql, db_path)
    utils.print_html("✅ Execute V2 query → Final answer:\n" + refined_df.to_html())
```

### 3.4 Run the SQL Workflow

Now it’s time for **you** to execute the complete SQL processing pipeline. This pipeline includes:

1. **Generate a SQL query** – The LLM creates a query from **your** question.
2. **Execute the query** – The query runs on the database.
3. **Evaluate the result** – The output is checked to confirm it answers **your** question.
4. **Refine the SQL with feedback** – If needed, the query is improved using feedback.
5. **Display the final output** – You see the refined query and its results.

#### Model Options

You can try different combinations of the following OpenAI models, each with different capabilities and performance:

* `openai:gpt-4o`
* `openai:gpt-4.1`
* `openai:gpt-4.1-mini`
* `openai:gpt-3.5-turbo`

💡 In this workflow, `openai:gpt-4.1` often gives the best results for self-reflection tasks.

**Important:** Because Large Language Models (LLMs) are stochastic, every run may return slightly different results.
You are encouraged to experiment with different models and combinations to find the setup that works best for **you**.


```python
run_sql_workflow(
    "products.db", 
    "Which color of product has the highest total sales?",
    model_generation="openai:gpt-4.1",
    model_evaluation="openai:gpt-4.1"
)
```

## 4. Final Takeaways

By completing this lab, **you** learned how to:

* Use an LLM to turn natural-language questions into SQL queries.
* Apply **reflection patterns** (external and self-feedback) to evaluate and refine your queries.
* Automate a complete SQL workflow, from schema extraction to query refinement.
* Experiment with different LLM models to compare performance and accuracy.

The key insight is that reflection makes **your** agent more reliable: instead of stopping at the first attempt, the agent can review, improve, and deliver results that better match **your** intent.


<div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">

🎉 <strong>Congratulations!</strong>  

You have completed the lab on building an **agentic SQL workflow**.  
Along the way, **you** practiced how planning, reflection, execution, and validation come together into a reliable pipeline.  

You also saw how reusing context makes tool usage more efficient, how validations improve safety, and how separating responsibilities across roles increases transparency and auditability.  

With these skills, **you** are ready to design your own agentic pipelines — pipelines that query data automatically while giving you confidence in their safety, explainability, and adaptability. 🌟  

</div>


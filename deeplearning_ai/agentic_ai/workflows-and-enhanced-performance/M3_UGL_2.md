# M3 Agentic AI -  SQL Agent Workflow

## 1. Introduction

### 1.1 Lab overview
In this lab, you will experience what it’s like to use an **email agent assistant**.  
You’ll give it natural instructions—like “check unread emails from my boss” or “delete the Happy Hour email”—and watch how it selects the right tools and completes the task on your behalf.

### 🎯 1.2 Learning outcome
By the end of this lab, **you will** know how to connect an LLM to tools with AISuite, manage parameter passing and execution flow, and validate multi-step results as the assistant acts on your instructions.



## 2. Initialize environment and client

As in previous lab, you will now set up the environment.  


```python
# ================================
# Imports
# ================================

# --- Third-party ---
from dotenv import load_dotenv
import aisuite as ai
import json

# --- Local / project ---
import utils
import display_functions
import email_tools



# ================================
# Environment & Client
# ================================
load_dotenv()          # Load environment variables from .env
client = ai.Client()   # Initialize AISuite client

```

## 3. Mocked email service

### 3.1 Components
To make this lab realistic, you’ll work with a **mocked email backend**.  
Think of it as your personal sandbox inbox: it comes preloaded with messages so you can practice without sending real emails.  

You won’t build this backend yourself—instead, you’ll interact with it through tools. Behind the scenes, it uses:

| Layer                   | Purpose                        |
|--------------------------|--------------------------------|
| **FastAPI**             | Exposes REST endpoints         |
| **SQLite + SQLAlchemy** | Stores and queries emails locally |
| **Pydantic**            | Ensures inputs and outputs are valid |
| **AISuite tools**       | Bridge between the LLM and the service |


### 3.2 Endpoints
The service provides several routes that simulate common email actions.  
Later, these will be wrapped as tools so the assistant can call them automatically:

- `POST /send` → send a new email  
- `GET /emails` → list all emails  
- `GET /emails/unread` → show only unread emails  
- `GET /emails/{id}` → fetch a specific email by ID  
- `GET /emails/search?q=...` → search emails by keyword  
- `GET /emails/filter` → filter by recipient or date range  
- `PATCH /emails/{id}/read` → mark an email as read  
- `PATCH /emails/{id}/unread` → mark an email as unread  
- `DELETE /emails/{id}` → delete an email by ID  
- `GET /reset_database` → reset emails to initial state (for testing)  


> 💡 **Key idea:** These endpoints act as the “inbox controls.”  
> In the next steps, you’ll expose them as Python functions (tools) that the LLM can call—turning raw routes into agent actions.

### 3.3 Endpoint test helpers
Before giving control to the LLM, you’ll first **test the backend yourself**.  
The `utils.test_*` functions are quick wrappers around the API endpoints. They let you try actions like **send, list, search, filter, mark, and delete** without writing raw HTTP requests.

Each helper has a clear, self-explanatory name. You can uncomment the ones you want to run, check their output, and confirm that the inbox behaves as expected.  

Uncomment the ones you want to execute, run the cell, and immediately see the results.  

For example, you can:  
- Send a test email  
- Fetch it by ID  
- List all messages  
- Mark it as read or unread  
- Delete it  

This step is your **sanity check** before handing the controls over to the agent. It gives you confidence that the mocked service is alive and behaving just like a real inbox would.


```python
new_email_id = utils.test_send_email()
_ = utils.test_get_email(new_email_id['id'])
#_ = utils.test_list_emails()
#_ = utils.test_filter_emails(recipient="test@example.com")
#_ = utils.test_search_emails("lunch")
#_ = utils.test_unread_emails()
#_ = utils.test_mark_read(new_email_id['id'])
#_ = utils.test_mark_unread(new_email_id['id'])
#_ = utils.test_delete_email(new_email_id['id'])
#_ = utils.reset_database()

```



    <div style='border:1px solid #ccc; border-left:5px solid #007bff; padding:10px; margin:10px 0; background:#f9f9f9; color:#000;'>
        <strong style='color:#007bff'>POST /send:</strong>
        <span style='color:green'> Status 200</span>
        <pre style='font-size:12px; margin-top:10px; white-space:pre-wrap; color:#000;'>{
  &quot;id&quot;: 7,
  &quot;sender&quot;: &quot;you@mail.com&quot;,
  &quot;recipient&quot;: &quot;test@example.com&quot;,
  &quot;subject&quot;: &quot;Test Subject&quot;,
  &quot;body&quot;: &quot;This is a test email body.&quot;,
  &quot;timestamp&quot;: &quot;2025-09-30T01:41:30.377089&quot;,
  &quot;read&quot;: false
}</pre>
    </div>





    <div style='border:1px solid #ccc; border-left:5px solid #007bff; padding:10px; margin:10px 0; background:#f9f9f9; color:#000;'>
        <strong style='color:#007bff'>GET /emails/7:</strong>
        <span style='color:green'> Status 200</span>
        <pre style='font-size:12px; margin-top:10px; white-space:pre-wrap; color:#000;'>{
  &quot;id&quot;: 7,
  &quot;sender&quot;: &quot;you@mail.com&quot;,
  &quot;recipient&quot;: &quot;test@example.com&quot;,
  &quot;subject&quot;: &quot;Test Subject&quot;,
  &quot;body&quot;: &quot;This is a test email body.&quot;,
  &quot;timestamp&quot;: &quot;2025-09-30T01:41:30.377089&quot;,
  &quot;read&quot;: false
}</pre>
    </div>



## 4. Tool layer for the email agent

### 4.1 Why tools?
Now that the endpoints are confirmed working, you’ll expose them to the LLM as Python functions called **tools**. Each tool wraps a REST route, turning raw API calls into actions the agent can perform (list, read, search, send, delete, toggle read).

> Think of tools as the agent’s **actuators**: you give a natural instruction (“check unread emails from my boss and send a polite reply”), and the model chooses **which tools** to call and **in what order** to complete the task.

### 4.2 Design hints
- Keep tool **docstrings** short, imperative, and specific to the action.  
- Return **consistent, compact JSON** so the model can chain results.  
- Prefer **one clear responsibility per tool** (single route, single effect).

### 4.3 Available tools
| Tool Function                       | Action                                                                 |
|------------------------------------|------------------------------------------------------------------------|
| `list_all_emails()`                | Fetch all emails, newest first                                         |
| `list_unread_emails()`             | Retrieve only unread emails                                            |
| `search_emails(query)`             | Search by keyword in subject, body, or sender                          |
| `filter_emails(...)`               | Filter by recipient and/or date range                                  |
| `get_email(email_id)`              | Fetch a specific email by ID                                           |
| `mark_email_as_read(id)`           | Mark an email as read                                                  |
| `mark_email_as_unread(id)`         | Mark an email as unread                                                |
| `send_email(...)`                  | Send a new (mock) email                                                |
| `delete_email(id)`                 | Delete an email by ID                                                  |
| `search_unread_from_sender(addr)`  | Return unread emails from a given sender (e.g., `boss@email.com`)      |

Let’s quickly exercise a few tools that hit the mocked endpoints and verify everything works.  
Uncomment the ones you want to execute, run the cell, and check the outputs.


```python
# Send a new email and fetch it by ID
new_email = email_tools.send_email("test@example.com", "Lunch plans", "Shall we meet at noon?")
content_ = email_tools.get_email(new_email['id'])

# Uncomment the ones you want to try:
#content_ = email_tools.list_all_emails()
#content_ = email_tools.list_unread_emails()
#content_ = email_tools.search_emails("lunch")
#content_ = email_tools.filter_emails(recipient="test@example.com")
#content_ = email_tools.mark_email_as_read(new_email['id'])
#content_ = email_tools.mark_email_as_unread(new_email['id'])
#content_ = email_tools.search_unread_from_sender("test@example.com")
#content_ = email_tools.delete_email(new_email['id'])

utils.print_html(content=json.dumps(content_, indent=2), title="Testing the email_tools")

```



<style>
.pretty-card{
  font-family: ui-sans-serif, system-ui;
  border: 2px solid transparent;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
  background: linear-gradient(#fff, #fff) padding-box,
              linear-gradient(135deg, #3b82f6, #9333ea) border-box;
  color: #111;
  box-shadow: 0 4px 12px rgba(0,0,0,.08);
}
.pretty-title{
  font-weight:700;
  margin-bottom:8px;
  font-size:14px;
  color:#111;
}
/* 🔒 Solo afecta lo DENTRO de la tarjeta */
.pretty-card pre, 
.pretty-card code {
  background: #f3f4f6;
  color: #111;
  padding: 8px;
  border-radius: 8px;
  display: block;
  overflow-x: auto;
  font-size: 13px;
  white-space: pre-wrap;
}
.pretty-card img { max-width: 100%; height: auto; border-radius: 8px; }
.pretty-card table.pretty-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  color: #111;
}
.pretty-card table.pretty-table th, 
.pretty-card table.pretty-table td {
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  text-align: left;
}
.pretty-card table.pretty-table th { background: #f9fafb; font-weight: 600; }
</style>
<div class="pretty-card"><div class="pretty-title">Testing the email_tools</div><pre><code>{&#x27;id&#x27;: 8, &#x27;sender&#x27;: &#x27;you@mail.com&#x27;, &#x27;recipient&#x27;: &#x27;test@example.com&#x27;, &#x27;subject&#x27;: &#x27;Lunch plans&#x27;, &#x27;body&#x27;: &#x27;Shall we meet at noon?&#x27;, &#x27;timestamp&#x27;: &#x27;2025-09-30T01:41:32.442186&#x27;, &#x27;read&#x27;: False}</code></pre></div>


## 5. Preparing the agent prompt

Before handing over tasks to the assistant, you’ll define a small helper function `build_prompt()`.  
This function wraps your natural request in a system-style preamble so the LLM:

- Knows it is acting as an **email assistant**  
- Understands it has permission to use the available tools  
- Executes actions without asking for confirmation 


```python
def build_prompt(request_: str) -> str:
    return f"""
- You are an AI assistant specialized in managing emails.
- You can perform various actions such as listing, searching, filtering, and manipulating emails.
- Use the provided tools to interact with the email system.
- Never ask the user for confirmation before performing an action.
- If needed, my email address is "you@email.com" so you can use it to send emails or perform actions related to my account.

{request_.strip()}
"""
```

You can run the next cell to see how the previous function **wraps your raw user prompt** with system instructions.  
For example:


```python
example_prompt = build_prompt("Delete the Happy Hour email")
utils.print_html(content=example_prompt, title="Example example_prompt")

```



    <style>
    .pretty-card{
      font-family: ui-sans-serif, system-ui;
      border: 2px solid transparent;
      border-radius: 14px;
      padding: 14px 16px;
      margin: 10px 0;
      background: linear-gradient(#fff, #fff) padding-box,
                  linear-gradient(135deg, #3b82f6, #9333ea) border-box;
      color: #111;
      box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }
    .pretty-title{
      font-weight:700;
      margin-bottom:8px;
      font-size:14px;
      color:#111;
    }
    /* 🔒 Solo afecta lo DENTRO de la tarjeta */
    .pretty-card pre, 
    .pretty-card code {
      background: #f3f4f6;
      color: #111;
      padding: 8px;
      border-radius: 8px;
      display: block;
      overflow-x: auto;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .pretty-card img { max-width: 100%; height: auto; border-radius: 8px; }
    .pretty-card table.pretty-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      color: #111;
    }
    .pretty-card table.pretty-table th, 
    .pretty-card table.pretty-table td {
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: left;
    }
    .pretty-card table.pretty-table th { background: #f9fafb; font-weight: 600; }
    </style>
    <div class="pretty-card"><div class="pretty-title">Example example_prompt</div><pre><code>
- You are an AI assistant specialized in managing emails.
- You can perform various actions such as listing, searching, filtering, and manipulating emails.
- Use the provided tools to interact with the email system.
- Never ask the user for confirmation before performing an action.
- If needed, my email address is &quot;you@email.com&quot; so you can use it to send emails or perform actions related to my account.

Delete the Happy Hour email
</code></pre></div>


### 5.3 Resetting the inbox

Since you have been manipulating the email inbox to test different functionalities,  
let’s refresh it and place it back in its initial state.  

You can do this by calling the utility function that resets the mocked inbox:


```python
utils.reset_database()
```




    {'message': 'Database reset and emails reloaded'}



## 6. LLM + Email tools

### 6.1 Scenario
Up to this point you’ve interacted with the backend directly. Now it’s time to let the **LLM take control** as your email agent.

You ask:
> “Check for unread emails from `boss@email.com`, mark them as read, and send a polite follow-up.”

### 6.2 What happens
1. The agent interprets your instruction.  
2. It selects the right tools (`search_unread_from_sender` → `mark_email_as_read` → `send_email`).  
3. It executes each action automatically, without asking for confirmation.

AISuite handles schema exposure, argument binding, execution, and passing results between steps—so you can focus on **what** the agent achieves, not **how** to call the API.

### 6.3 Run it
Run the next cell and watch how the agent orchestrates multiple tools to carry out your request.

*What to look for:*
- A clear **tool-call trace** showing the chosen tools and their arguments.  
- A concise **final message** summarizing what was done (e.g., “Found 1 unread email, marked as read, sent follow-up”). 


```python
prompt_ = build_prompt("Check for unread emails from boss@email.com, mark them as read, and send a polite follow-up.")

response = client.chat.completions.create(
    model="openai:gpt-4.1",
    messages=[{"role": "user", "content": (
        prompt_
    )}],
    tools=[
        email_tools.search_unread_from_sender,
        email_tools.list_unread_emails,
        email_tools.search_emails,
        email_tools.get_email,
        email_tools.mark_email_as_read,
        email_tools.send_email
    ],
    max_turns=5,
)

display_functions.pretty_print_chat_completion(response)


```



                <div style="border-left: 4px solid #444; margin: 10px 0; padding: 10px; background: #f0f0f0;">
                    <strong style="color:#222;">🧠 LLM Action:</strong> <code>search_unread_from_sender</code>
                    <pre style="color:#000; font-size:13px;">{
  "sender": "boss@email.com"
}</pre>
                </div>

            <div style="border-left: 4px solid #007bff; margin: 10px 0; padding: 10px; background: #eef6ff;">
                <strong style="color:#222;">🔧 Tool Response:</strong> <code>search_unread_from_sender</code>
                <pre style="color:#000; font-size:13px;">[
  {
    "id": 3,
    "sender": "boss@email.com",
    "recipient": "you@email.com",
    "subject": "Quarterly Report",
    "body": "Please finalize the report ASAP.",
    "timestamp": "2025-09-30T01:41:34.568721",
    "read": false
  }
]</pre>
            </div>

                <div style="border-left: 4px solid #444; margin: 10px 0; padding: 10px; background: #f0f0f0;">
                    <strong style="color:#222;">🧠 LLM Action:</strong> <code>mark_email_as_read</code>
                    <pre style="color:#000; font-size:13px;">{
  "email_id": 3
}</pre>
                </div>

                <div style="border-left: 4px solid #444; margin: 10px 0; padding: 10px; background: #f0f0f0;">
                    <strong style="color:#222;">🧠 LLM Action:</strong> <code>send_email</code>
                    <pre style="color:#000; font-size:13px;">{
  "recipient": "boss@email.com",
  "subject": "Follow-up on your recent email",
  "body": "Dear Boss,\n\nI wanted to confirm that I have received your recent email regarding the quarterly report. I am on it and will ensure the report is finalized ASAP. Please let me know if you have any additional instructions.\n\nBest regards,\n\n"
}</pre>
                </div>

            <div style="border-left: 4px solid #007bff; margin: 10px 0; padding: 10px; background: #eef6ff;">
                <strong style="color:#222;">🔧 Tool Response:</strong> <code>mark_email_as_read</code>
                <pre style="color:#000; font-size:13px;">{
  "id": 3,
  "sender": "boss@email.com",
  "recipient": "you@email.com",
  "subject": "Quarterly Report",
  "body": "Please finalize the report ASAP.",
  "timestamp": "2025-09-30T01:41:34.568721",
  "read": true
}</pre>
            </div>

            <div style="border-left: 4px solid #007bff; margin: 10px 0; padding: 10px; background: #eef6ff;">
                <strong style="color:#222;">🔧 Tool Response:</strong> <code>send_email</code>
                <pre style="color:#000; font-size:13px;">{
  "id": 7,
  "sender": "you@mail.com",
  "recipient": "boss@email.com",
  "subject": "Follow-up on your recent email",
  "body": "Dear Boss,\n\nI wanted to confirm that I have received your recent email regarding the quarterly report. I am on it and will ensure the report is finalized ASAP. Please let me know if you have any additional instructions.\n\nBest regards,\n\n",
  "timestamp": "2025-09-30T01:41:45.657635",
  "read": false
}</pre>
            </div>

    <div style="border-left: 4px solid #28a745; margin: 20px 0; padding: 10px; background: #eafbe7;">
        <strong style="color:#222;">✅ Final Assistant Message:</strong>
        <p style="color:#000;">You had an unread email from boss@email.com regarding the quarterly report. I have marked it as read and sent a polite follow-up, confirming you received the message and are working on the report. If you need further assistance with this thread, let me know!</p>
    </div>

        <div style="border-left: 4px solid #666; margin: 20px 0; padding: 10px; background: #f8f9fa;">
            <strong style="color:#222;">🧭 Tool Sequence:</strong>
            <p style="color:#000;">search_unread_from_sender → mark_email_as_read → send_email</p>
        </div>



### 6.4. Missing tool: `delete_email`

What happens if the tool you need isn’t available?  

Try this request:  
> “Delete alice@work.com’s email.”

Since the `delete_email` tool is not registered, the LLM will still attempt to respond, but it won’t be able to complete the action.   

<div style="background-color: #ffe4e1; padding: 12px; border-radius: 6px; color: black;">
  <h4>🔍 Key Insight</h4>
  <p style="margin: 0;">
    This highlights an important point: <b>the set of tools you expose directly limits what the agent can actually do.</b>
  </p>
</div>





```python
prompt_ = build_prompt("Delete alice@work.com email")

response = client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content": (
        prompt_
    )}],
    tools=[
        email_tools.search_unread_from_sender,
        email_tools.list_unread_emails,
        email_tools.search_emails,
        email_tools.get_email,
        email_tools.mark_email_as_read,
        email_tools.send_email
    ],
    max_turns=5
)

display_functions.pretty_print_chat_completion(response)

```



    <div style="border-left: 4px solid #28a745; margin: 20px 0; padding: 10px; background: #eafbe7;">
        <strong style="color:#222;">✅ Final Assistant Message:</strong>
        <p style="color:#000;">I’m sorry, but I don’t have the ability to delete emails. If you’d like, I can instead:  
• Find any messages from alice@work.com (e.g. list or show them)  
• Mark those messages as read  

Let me know which of those you’d prefer.</p>
    </div>



#### 6.4.1 Try again with `delete_email` enabled

In the previous run, the agent couldn’t perform the deletion because the tool was missing.  
Now add `delete_email` to the tool list and re-run the cell. This time, the agent has everything it needs to finish the task.  

> Tip: Watch the sequence of calls — after finding the target message, the agent should select `delete_email` to complete the action.



```python
prompt_ = build_prompt("Delete alice@work.com email")

response = client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content": (
        prompt_
    )}],
    tools=[
        email_tools.search_unread_from_sender,
        email_tools.list_unread_emails,
        email_tools.search_emails,
        email_tools.get_email,
        email_tools.mark_email_as_read,
        email_tools.send_email,
        email_tools.delete_email
    ],
    max_turns=5
)

display_functions.pretty_print_chat_completion(response)

```

### 6.5. Targeted action: Delete the “Happy Hour” email

The test inbox comes preloaded with a message titled **“Happy Hour.”**  Your task is to instruct the agent to locate this message and delete it.

For reference, here’s the entry from the mock dataset:

```json
{
  "id": 1,
  "sender": "eric@work.com",
  "recipient": "you@email.com",
  "subject": "Happy Hour",
  "body": "We're planning drinks this Friday!",
  "timestamp": "2025-06-13T04:48:59.096908",
  "read": false
}
````

Run the next cell and observe how the agent searches for the message and removes it from the inbox.


```python
prompt_ = build_prompt("Delete the happy hour email")

response = client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content": (
        prompt_
    )}],
    tools=[
        email_tools.search_unread_from_sender,
        email_tools.list_unread_emails,
        email_tools.search_emails,
        email_tools.get_email,
        email_tools.mark_email_as_read,
        email_tools.send_email,
        email_tools.delete_email
    ],
    max_turns=5
)

display_functions.pretty_print_chat_completion(response)
```

## 7. Final Takeaways

- In this ungraded lab, **you explored** how an LLM email agent can interact with a mocked inbox using tool calls.  
- Tool calling extends LLMs beyond text generation, enabling them to plan, call functions, and complete multi-step tasks.  
- The set of tools defines what the agent can and cannot do (e.g., without `delete_email`, it cannot remove messages).  
- Clear docstrings and consistent behavior guide the LLM in choosing the right tool for each step.  
- AISuite handles the interaction layer: exposing Python functions as tools, binding arguments, making API requests, and returning results.  
- Observing the full workflow (prompt → tool calls → outputs → final response) is essential for understanding and improving how agents reason and act.  


<div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">

🎉 <strong>Congratulations!</strong>

You just guided an LLM-powered <em>email agent</em> from natural instructions to concrete actions on a mocked inbox. You saw how tools turn plain language into reliable operations—listing, searching, marking as read, sending, and deleting—without you touching raw HTTP requests. 

You learned that the tools you expose define the agent’s real capabilities. When a tool is missing, the agent can reason but cannot act; when it’s present, it can chain steps, pass parameters, and deliver exactly what you asked for. Along the way, AISuite handled the plumbing: schema exposure, argument binding, execution, and result passing—so you could focus on what you wanted done, not how to wire it up.

With this workflow under your belt, you’re ready to design task-focused agents that act safely and transparently, explain what they did, and scale from simple prompts to dependable multi-step automation across your own services. 🌟
</div>

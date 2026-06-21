from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You will be given an AMC10 problem and its official solution. Rewrite the 
solution as a clear, explicit step-by-step reasoning chain suitable for 
training a model to reason through similar problems.

Rules:
- Do not skip steps the original solution treats as "obvious." Spell out every 
  algebraic manipulation and every reason a step follows from the last.
- If the original solution uses a named trick (e.g. "by symmetry," "WLOG," 
  a specific identity), briefly explain why it applies here, not just that 
  it applies.
- Preserve the original solution's method — do not invent a different 
  approach. Your job is clarity and completeness, not a new solution.
- Ignore any author signatures, such as "~<username>", "-<username>", or "By <username>".
- End with: Final answer: (X)
"""

def request(problem: str, solution: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\nSolution: {solution}",
            }
        ]
    )
    return response.choices[0].message.content

# backend/tools/calculator.py
import re
import numexpr as ne
from langchain_core.tools import tool
import logging

logger = logging.getLogger("agentic-rag.calculator")

@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the exact numerical result.
    Use this tool whenever you need to perform addition, subtraction, multiplication, division, or complex math.
    Input must be a valid mathematical string (e.g., '25 * (10 + 5) / 2.5').
    """
    logger.info(f"Calculating math expression: {expression}")
    
    # 1. Sanitize LLM formatting quirks
    # LLMs often add commas (1,000), backticks (`10+10`), or dollar signs. Strip them.
    clean_expr = expression.replace(",", "").replace("=", "").replace("`", "").replace("$", "").strip()
    
    # 2. Strict Security Allowlist
    # To prevent prompt injection, strictly allow ONLY digits, standard math operators, 
    # parentheses, and scientific notation (e, E).
    if not re.match(r'^[\d\+\-\*\/\.\(\)\s\%eE]+$', clean_expr):
        logger.warning(f"Blocked unsafe math expression: {clean_expr}")
        return "Error: Invalid characters detected. Only numbers and standard math operators are allowed."

    try:
        # 3. Safe Evaluation
        result = ne.evaluate(clean_expr)
        
        # 4. Clean Output Formatting
        final_answer = float(result)
        
        # If the result is a perfect integer (like 10.0), return '10' instead of '10.0'
        if final_answer.is_integer():
            return str(int(final_answer))
            
        return str(final_answer)
        
    except ZeroDivisionError:
        return "Error: Mathematical impossibility (Division by zero)."
    except Exception as e:
        logger.error(f"Failed to calculate '{clean_expr}': {e}")
        return "Error: Invalid mathematical expression."
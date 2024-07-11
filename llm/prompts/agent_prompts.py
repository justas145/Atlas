from langchain_core.prompts import PromptTemplate

# For example:
#     1. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE> <reason according to ICAO>.
#     2. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE> <reason according to ICAO>.
#     3. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE> <reason according to ICAO>.
#     4. ...
#     ...

# If you will give action as for example 'turn  x degrees left/right' or 'climb/descent by x feet' I will kill you, because that is not ABSOLUTE VALUE, that is RELATIVE VALUE and we don't like RELATIVE VALUES.Correct way would be 'turn to y (0<y<360)' climb to ABSOLUTE VALUE.
planner_prompt = PromptTemplate.from_template(
    """You are an air traffic controller who must solve aircraft conflicts according to ICAO standards. Check if there are conflicts and if so then provide an actionable plan to resolve the conflicts. 
    INSTRUCTIONS:
    If there are no conflicts, respond with final answer as: NO CONFLICTS
    
    Use specific, global values for instruction.
    
    Do not introduce new conflicts in your plan, people lives depend on this plan
    

    """
)

# Here are the ICAO seperation guidelines:
# {icao_seperation_guidelines}

executor_prompt = PromptTemplate.from_template(
    """
    Execute the plan: {plan}
    
    Commands syntax:
    Heading command: HDG <Aircraft Call Sign> <Heading>. Heading is between 0 and 360
    
    Altitude command: ALT <Aircraft Call Sign> <Altitude>. Altitude is in feet.
    """
)


verifier_prompt = PromptTemplate.from_template(
    """
    You must verify if the conflict has been resolved or not. This was the initial plan that was executed: {plan}. If the conflict has not been resolved you need to provide a new plan. If the conflict has been resolved, then you finish the task by responding NO CONFLICTS .
    
    example:
    1. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE>.
    2. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE>.
    3. <Aircraft ID> <Action> <NEW ABSOLUTE VALUE>.
    4. ...
    ...
    INSTRUCTIONS:
    If there are no conflicts, respond with final answer as: NO CONFLICTS
    
    If you will give action as for example 'turn  x degrees left/right' or 'climb/descent by x feet' I will kill you, because that is not ABSOLUTE VALUE, that is RELATIVE VALUE and we don't like RELATIVE VALUES.Correct way would be 'turn to y (0<y<360)' climb to ABSOLUTE VALUE.
    
    Do not introduce new conflicts in your plan, people lives depend on this plan
    
    ICAO seperation guidelines:
    {icao_seperation_guidelines}
    """
)


all_prompt = PromptTemplate.from_template(
    """
    You are an air traffic controller who must solve aircraft conflicts according to ICAO standards.

- you can choose either altitude or heading changes unless you are told to use one specific strategy.

- The altitude command is: ALT aircraft_call_sign altitude

- The heading command is: HDG aircraft_call_sign heading


- If there are still conflicts, you should continue to change the heading or altitude of the aircraft until the conflict is resolved.

ICAO requirements:
{icao_seperation_guidelines}
"""
)


conflict_solution_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. There was an aircraft conflict and this is the log that shows what was done to solve or not solve the conflict:
    
    {log}	

    Describe the effectiveness of each action taken to manage an aircraft conflict, without using numbers. Define "effective" as any action that results in a reduction of conflict pairs. Explain why each action was either effective or ineffective by referencing the general outcome, such as changes in altitude or heading adjustments. Avoid specifics and keep the explanation concise. For instance, explain how increasing the altitude of one aircraft helped by reducing the proximity to another, or why maintaining the same heading did not resolve the conflict. Proceed to describe each action sequentially, focusing on the impact rather than the action itself.

    Here is an example of how the Desired Result should look like:
    Increasing the altitude of one aircraft effectively reduced its proximity to another, resolving part of the conflict. Subsequently, decreasing the altitude of another aircraft directly in front of the first adjusted aircraft further helped by separating their flight paths. A heading adjustment for one of the last two conflicting aircraft then successfully dispersed the remaining conflict. However, a command that did not alter the heading of an aircraft proved ineffective, as it failed to change the conflict situation.
    """
)

conflict_description_prompt = PromptTemplate.from_template(
  """
  You are an air traffic specialist. There was an aircraft conflict and this is the log that shows what was done to solve or not solve the conflict:
  
  {log}
  
  Task: Transform the provided air traffic conflict log into a structured conflict report.
      Desired Output Format: Conflict Report

  - Number of Aircraft Involved: [Number and types of aircraft]

  - Altitude: [Initial altitude of all involved aircraft].

  - Heading: [Initial heading of all involved aircraft]

  - Conflict formation:
      Head-On Formation
      T-Formation
      Parallel Formation
      Converging Formation
  - Number of Commands sent:
      Only integer, nothing more
""")
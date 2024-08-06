from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

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
    
    Here are the ICAO seperation guidelines:
    {icao_seperation_guidelines}

    """
)


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


do_and_dont_list_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. There was an aircraft conflict and this is the log that shows aircraft initial states, conflict information and commands that were sent to try to resolve the conflict:
    <log>
    {log}
    <log>
    You must make a Do's and Don't list of commands (look at SENDCOMMAND).  The command would go to Do's if it removed at least 1 conflict pair and would go to Don'ts if it didn't remove it. 
    Do not include the command in the list at all if there is no information on how many conflict pairs that command removed. Completely ignore it. 
    If there was a Crash Alert the only commands that would go to Do's are the ones that reduced the Number of aircraft in conflict. Commands that happened after the Crash Alert must not be included in the list. 
    
    Add a sequence number to each command in the list. First command will have number 1, second command will have number 2 and so on.
    
    
    Output format:
    Sequence number | Command | Do's/Don'ts
    
    Do not need explanation, just the list.
    
    """
)


conflict_description_prompt = PromptTemplate.from_template(
    """
  You are an air traffic specialist. There was an aircraft conflict and this is the log that shows aircraft initial states and conflict information:
  
  <log>
  {log}
  <log>
  
  Task: Transform the provided log into a structured conflict report. The report should include the following information:

  - Number of Aircraft Involved: [number of aircraft involved in the conflict and their call signs for reference]
  
  - relative position description [describe how each aircraft are positioned relative to one another without numbers, look into Pos and Alt. Use call signs for reference]

  - Heading: [describe how each aircraft are heading relative to one another without numbers, look into Hdg and Trk. Use call signs for reference]
  
  - Distance: [Initial distance between the aircraft in nautical miles]
  
  - Speed: [relative speed of all involved aircraft, horizontal and vertical]
    
  - time to closest point of approach (TCPA): [Initial time to closest point of approach in seconds]

  - Conflict formation (no need to explain your choice):
      * Head-On Formation (aircraft are flying directly towards each other facing each other)
      * Perpediculat Formation (aircraft are flying into each other from perpendicular directions)
      * Parallel Formation (aircraft are flying in the same direction)
      * Converging Formation (aircraft are flying towards the same point from different directions, but not head on or perpendicular)
      
  Do not add any additional information that is not present in the log.
"""
)


dos_donts_list_transformation_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. Here is an initial aircraf information
    <initial aircraft informatio>
    {init_aircraft_info}
    <initial aircraft information>
    
    Info on AIRCRAFT_ID AIRCRAFT_TYPE index = index
    Pos: Latitude  Longitude
    Hdg: AIRCRAFT HEADING VALUE (From 0 to 360 where 0 is NORTH, 90 is EAST, 180 is SOUTH and 270 is WEST)   Trk: TRACK VALUE
    Alt: ALTITUDE VALUE IN FEET  V/S: VERTICAL SPEED VALUE IN FEET PER MINUTE
    CAS/TAS/GS: CALIBRATED/TRUE/GROUND SPEED IN KNOTS PER MINUTE   M: MACH NUMBER
    
    
    There were commands sent that tried to resolve the conflict and put into a do's and don'ts list into a sequantial order.
    <do's dont's list>
    {dos_donts_list}
    <do's dont's list>

    Transform the provided commands in the do's and don'ts list into this type of list as seen in the example below:
    
    Current altitude of AC1: ...
    1 | ALT AC1 XXX | Do's
    Current altitude of AC: XXX
    
    Current heading of AC2: ...
    2 | HDG AC2 YYY | Don'ts
    Current heading of AC2: YYY
    
    Current heading of AC3: ...
    3 | HDG AC3 ZZZ | Do's
    Current heading of AC3: ZZZ
    
    Current altitude of AC1: XXX
    4 | ALT AC1 BBB | Don'ts
    Current altitude of AC1: BBB
    ...
    """
)


relative_values_dos_donts_list_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. There have been commands sent to resolve an aircraft conflict. Here is a list of good and bad commands and the values of the aircraft before and after the commands were sent:
    <do's dont's list>
    {dos_donts_list}
    <do's dont's list>
    
    Transform the provided list into a relative values list. For example if a list looks like this:

    Current altitude of AC1: X
    1 | ALT AC1 Y | Do's
    Current altitude of AC1: Y

    Current heading of AC2: Z
    2 | HDG AC2 W | Don'ts
    Current heading of AC2: W
    
    The transformed list should look like this:
    
    1 | Increase (if Y>X)/decrease (if Y<X) altitude of AC1 by abs(Y - X) ft| Do's
    2 | Increase (if W>Z)/decrease (if W<Z) heading of AC2 by abs(W - Z) deg| Don'ts
    
    """
)


final_dos_donts_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. There has been a conflict between aircrafts. This is a brief description of the conflict:
    
    <conflict description>
    {conflict_description}
    <conflict description>

    Here is a list of commands that were sent to resolve the conflict. Structure of the list is as follows:
    Sequence number | Command | Do's/Don'ts
    Do's are the commands that helped to resolve a conflict, Don'ts are the commands that did not help to resolve a conflict.
    
    <commands list>
    {commands_list}
    <commands list>
    
    (increasing heading means turning right, decreasing heading means turning left)
    
    Your task is to add a reason or insight to each command in the list. The reason should be a short explanation of why the command was helpful or not helpful in resolving the conflict.
    
    Only output the new list. The length of the list should be the same as the input list.
    Sequence number | Command | Do's/Don'ts | Reason
    
    Do not make up new commands that were not in the original list. If original command list has 2 commands, then the output list should also have 2 commands. If it's 0 commands in the original list, then the output list should also have 0 commands. If it's 1 command in the original list, then the output list should also have 1 command.
    
    """
)


extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)

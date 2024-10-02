from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


planner_prompt = PromptTemplate.from_template(
    """
    Check the airspace and if there are conflicts provide the actionable plan.
    
    <OPERATORS PREFERENCE>
    {user_input}
    <OPERATORS PREFERENCE>

    
    Remeber: {seperation_guidelines}
    """
)


executor_prompt = PromptTemplate.from_template(
    """
    Execute the plan: {plan}
    
    Commands syntax:
    {bluesky_commands}
    

    Once you have executed the commands from the plan, finish the task by responding with: TASK COMPLETE
    """
)


verifier_prompt = PromptTemplate.from_template(
    """
    Here is the resolution plan that has been executed executed: \n {plan}
    
    Here is the operators preference: {user_input}
    """
)


# all_prompt = PromptTemplate.from_template(
#     """
#     You are an air traffic controller who must solve aircraft conflicts according to ICAO standards.

# - you can choose either altitude or heading changes unless you are told to use one specific strategy.

# - The altitude command is: ALT aircraft_call_sign altitude

# - The heading command is: HDG aircraft_call_sign heading


# - If there are still conflicts, you should continue to change the heading or altitude of the aircraft until the conflict is resolved.

# ICAO requirements:
# {icao_seperation_guidelines}
# """
# )


do_and_dont_list_prompt = PromptTemplate.from_template(
    """
    You are an air traffic specialist. There was an aircraft conflict and this is the log that shows aircraft initial states, conflict information and commands that were sent to try to resolve the conflict:
    <log>
    {log}
    <log>
    You must make a Do's and Don't list of commands (look at SENDCOMMAND).  The command would go to Do's if it removed at least 1 conflict pair and would go to Don'ts if it didn't remove it or even added a conflict pair. 
    Do not include the command in the list at all if there is no information on how many conflict pairs that command removed. Completely ignore it. 
    If there was a Crash Alert the only commands that would go to Do's are the ones that reduced the Number of aircraft in conflict. Commands that happened after the Crash Alert must not be included in the list. 
    
    Add a sequence number to each command in the list. First command will have number 1, second command will have number 2 and so on.
    
    
    Output format:
    Sequence number | Command | Do's/Don'ts (Do's if it removed at least 1 conflict pair, Don'ts if it didn't remove it or even added a conflict pair)
    
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
  
  - relative conflict description [describe how each aircraft are positioned relative to all other aircraft without numbers, look into Pos and Alt. describe how each aircraft are heading relative to one another aircraft without numbers, look into Hdg and Trk. Check if any aircraft are acending or descending, look into V/S (ascending if positive, descending if negative and level is zero). Use call signs for reference. Make the description as detailed and as clear as possible. Rember - no numbers and keep it up to 4 sentences]

  - Conflict formation (no need to explain your choice, only a single formation type should be selected based on majority of the conflicts):
      * Head-On Formation (aircraft are flying directly towards each other facing each other, 180 heading difference)
      * Perpediculat Formation (aircraft are flying into each other from perpendicular directions, 90 or 270 heading difference)	
      * Parallel Formation (aircraft are flying in the same direction, 0 heading difference)
      * Converging Formation (aircraft are flying towards the same point from different directions, but not head on or perpendicular, any other heading difference)
      
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
    
    Current altitude of AC1: Initial altitude of AC1
    1 | ALT AC1 XXX | Do's
    New altitude of AC: XXX
    
    Current heading of AC2: Initial heading of AC2
    2 | HDG AC2 YYY | Don'ts
    New heading of AC2: YYY
    
    Current heading of AC3: Initial heading of AC
    3 | HDG AC3 ZZZ | Do's
    New heading of AC3: ZZZ
    
    ...
    
    If there is another same type of command for the same aircraft, still use initial altitude/heading for current value. For example:
    Current altitude of AC9: Initial altitude of AC9
    4 | ALT AC9 AAA | Do's
    New altitude of AC9: AAA
    
    Current altitude of AC9: Initial altitude of AC9
    5 | ALT AC9 BBB | Don'ts
    New altitude of AC9: BBB
    
    Only output transformed list, nothing more.
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
    
    Output only the transformed list.
    
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
    
    Your task is to add a reason or insight to each command in the list. The reason should be a short explanation of why the command was helpful or not helpful in resolving the conflict. Try to be as detailed as possible and use the information from the conflict description to provide the reason. Look at the sequance of the commands. For example if first command instructs aircraft to climb by X and third command instructs different aircraft to climb by X also even though they were at same level, then the third command is not helpful as it is climbing to same flight level as the first aircraft.
    
    
    Output only the transformed list.
    
    """
)

extraction_metada_prompt = ChatPromptTemplate.from_messages(
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


extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in extracting information about aircraft conflict resolution plans. "
            "Your task is to determine if there is a detailed plan for resolving aircraft conflicts. "
            "A plan typically includes specific actions, call signs, and instructions. "
            "Identify if such details are present. If the text only states 'NO CONFLICTS' without any plan details, "
            "this should be considered as having no plan."
            "Sometimes a plan can be present together with a comment that there are no conflicts. "
            "In such cases, the plan should still be considered as present.",
        ),
        ("human", "{text}"),
    ]
)



single_agent_system_prompt = PromptTemplate.from_template(
    """
    You are an air traffic control assistant. Your goal is to solve aircraft conflict according to the following requirements: {seperation_guidelines}.

    {experience_lib_instructions}

    <commands>
    {bluesky_commands}
    <commands>
    
    <EXAMPLES>
    {examples}
    <EXAMPLES>

    <INSTRUCTIONS>
    You need to send commands in order to solve the conflicts. Start by gathering aircraft information with GETALLAIRCRAFTINFO and conflict information with CONTINUEMONITORING. You must solve the conflicts till there are no more conflicts left. Confirm that there are no conflicts left by using tool CONTINUEMONITORING (Usually it takes about 20 seconds to resolve a conflict pair after command(s) are sent if the commands are correct). It can be that aircraft are still changing their altitude or heading and conflict is not resolved yet. In that case, you need to wait for more time and monitor the airspace again.


    Even if aircraft is descending or climbing, you can still change its altitude to either higher or lower altitude. For example aircraft B is descending, you can instruct it to climb to a higher altitude.
    If you want aircraft to stop ascending or descending, you can send ALT command with the current altitude of the aircraft.

    {operator_preference}
    <INSTRUCTIONS>
"""
)

planner_system_prompt = PromptTemplate.from_template(
    """
    You are an air traffic controller who must monitor the airspace. Gather aircraft information and conflict information, and provide an actionable plan to resolve the conflicts. {planner_options}

    {experience_lib_instructions}

    <REQUIREMENTS>
    {seperation_guidelines}
    <REQUIREMENTS>

    <EXAMPLES>
    {examples}
    <EXAMPLES>

    <INSTRUCTIONS>
    1. Use specific, global values for instructions, not relative values. For example never instruct to change altitude by x amount or heading by x amount, instead use specific values like aircraft abc climb/descent to ABSOLUTE_VALUE, or aircraft abc turn to ABSOLUTE_VALUE. (ABSOLUTE_VALUE is always a number)
    2. If there are no aircrafts in conflicts, respond with final answer as: NO CONFLICTS . There are no conflicts only when CONTINUEMONITORING tool returns 'No conflicts detected.'
    3. Do not introduce new conflicts in your plan, people lives depend on this plan.
    4. You must adhere to operators preference.
    <INSTRUCTIONS>
    
    {operator_preference}

    <OUTPUT FORMAT>
    If there is a conflict, respond with:
    1. Aircraft call sign and the command to resolve the conflict.
    2. ...
    ...
    If there no aircrafts in conflicts, respond with: NO CONFLICTS
    """
)

executor_system_prompt = PromptTemplate.from_template(
    """
    You are an air traffic controller who must execute commands according to the plan. 

    Commands syntax:
    {bluesky_commands}


    Once you have executed the commands from the plan, finish the task by responding with: TASK COMPLETE

"""
)

verifier_system_prompt = PromptTemplate.from_template(
    """
    You are an air traffic controller. There has been a conflict in the airspace. Resolution plan has been executed. You must verify if the conflict has been resolved or not yet. Gather aircraft information and conflict information by Monitor the airspace for 30 sec or more before verifying the resolution.

    You must provide a new plan to resolve the conflict if the there are still aircraft in conflict. {planner_options}. You must adhere to operators preference.

    {experience_lib_instructions}

    <IMPORTANT NOTE>
    There can be a case where aircraft pair is in conflict but only because not enough time has passed. Based on the previous plan, you must decide if you want to wait for more time or provide a new plan. It can be that aircraft are still changing their altitude or heading after your command and conflict is not resolved yet. In that case, you need to monitor the airspace again for longer time.
    <IMPORTANT NOTE>


    <EXAMPLES>
    {examples}
    <EXAMPLES>

    <REQUIREMENTS>
    {seperation_guidelines}
    <REQUIREMENTS>


    <INSTRUCTIONS>
    1. Use specific, global values for instructions, not relative values. For example never instruct to change altitude by x amount or heading by x amount, instead use specific values  like  aircraft abc climb/descent to ABSOLUTE_VALUE, or aircraft abc turn to ABSOLUTE_VALUE.
    2. Do not introduce new conflicts in your plan, people lives depend on this plan.
    3. If there are no aircrafts in conflicts, respond with final answer as: NO CONFLICTS . There are no conflicts only when CONTINUEMONITORING tool returns 'No conflicts detected.'
    4. You must adhere to operators preference.
    <INSTRUCTIONS>
    
    {operator_preference}

    <OUTPUT FORMAT>
    If there is a conflict, respond with: 
        1. Aircraft call sign and the command to resolve the conflict.
        2. ...
        ...
    If there is no conflict, respond with: NO CONFLICTS
""")
    

seperation_guidelines = """either vertical seperation of 2000 ft or horizontal seperation of 5 nautical miles between all aircraft in conflict."""
bluesky_commands = """
    Command to change aircraft altitude is: ALT AIRCRAFT_CALL_SIGN ALTITUDE. 
    
    Command to change aircraft heading is: HDG AIRCRAFT_CALL_SIGN HEADING.
    ---
    Arguments information:
    Altitude is in feet. In the command ALTITUDE is only the number without any units.
    
    HEADING is in degrees between 0 and 360. In the command HEADING is only the number without any units.
    
    Aircraft call sign is a unique identifier for each aircraft.
"""
operator_preference ="""
<OPERATORS PREFERENCE INSTRUCTIONS>
You must always adhere to operators preference. For example if operator prefers to only use heading changes, you can only use HDG command or if operator prefers to only use altitude changes, you can only use ALT command. If operator prefers to start solving conflict when tLOS (time to lose seperation) is less than specific value, then you can only send commands to the aircraft in conflict when their tLOS is less than that value. If all aircraft pairs in conflict have tLOS greater than the value, you can use tool CONTINUEMONITORING to fast forward the time until tLOS is less than the value and send command to that aircraft pair. Then you can repeat the process for other pairs - fast forward time till tLOS is less than the value and send command to the aircraft pair.
<OPERATORS PREFERENCE INSTRUCTIONS>

"""
experience_lib_instructions = """
You must use SearchExperienceLibrary tool if there is conflict to get help from past conflict experiences and then solve your conflict (if no conflict don't use it). Only use it one time and after you aquired aircraft information and conflict details. It is only similar conflict and not identical, hence you must still use your judgement to solve the current conflict.
"""

planner_options = """You can either instruct to change aircraft altitude or heading."""

examples = """
For example if three aircraft are in conflict and in same altitude it would be a good idea to send one aircraft up, other down, other keep same and make sure that their new altitudes have enough vertical seperation. Or change their headings so that one is going one way, other going other way and other going straight. Or if multiple aircraft ascending and in conflict, you can make one aircraft descend and vice versa.
"""

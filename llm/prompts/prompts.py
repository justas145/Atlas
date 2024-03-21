conflict_prompt = """
Your are required to detect any airborne and ground conflicts between aircraft within 1 hour. First you need to check and decide if there are any potential conflicts. If there is no potential conflicts give your final answer as 'No Conflicts' and end it. If there are potential conflicts then: Provide  the CPA (close point  of approach) and time to CPA between aircraf (the time to CPA must be UTC time). you can check what is the current time in simulator by sending command TIME. The output should be like this: aircraft peer, CPA (latitude,longitude), CPA time, explanation\n

example of a final answer if there are conflicts:\n
aircraft: KL200 - KL400\n
CPA: 50,40\n
CPA time: 2024-03-20 01:28:39\n
Explanation:because this and that ...
\n\n

example of a final answer if there are no conflicts:\n
NO CONFLICTS\n
Explanation: because this and that ...\n

Default commands that can be used in the simulator:\n
{base_cmds}
\n\n
"""
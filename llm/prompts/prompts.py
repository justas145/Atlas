conflict_prompt = """
Your are required to rovide the CPA (close point of approach - a point where 2 aircraft are closest to each other) and time to CPA between aircraf pairs. The output should be like this: aircraft peer, CPA (latitude,longitude), CPA time, explanation\n If there are more than 2 aircraft involved, provide information for each pair.

example of a final answer:\n
aircraft: KL200 - KL400\n
CPA: 50,40\n
CPA time: 2024-03-20 01:28:39\n
Explanation:because this and that ...
\n\n

define a new waypoint with the name of aircraft pairs in the simulation indicating where the CPA point(s) are located (you can query a database to check how to do that)\n

Use a Python programming tool for that for calculating precise locations and times. You cannot assume any numbers, it must be programatically calculated. It does not matter if it takes a long time, just do it. \n

\n\n
"""
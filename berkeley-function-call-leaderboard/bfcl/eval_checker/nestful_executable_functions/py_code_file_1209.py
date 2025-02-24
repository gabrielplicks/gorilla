from typing import Dict, List



def calc_money(people: List[Dict[str, int]]) -> Dict[str, int]:

    """Calculates the total amount of money each person has.



    Args:

        people: A list of dictionaries, each of which has a `name` and a `money` attribute.



    Returns:

        A dictionary of name-money pairs.

    """

    money = {}

    for person in people:

        money[person['name']] = person['money']

    return money



people = [

    {'name': 'Alice', 'money': 10},

    {'name': 'Bob', 'money': 20},

]

print(calc_money(people))


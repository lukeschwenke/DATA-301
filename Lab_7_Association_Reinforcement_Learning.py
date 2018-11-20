import pandas as pd
from apyori import apriori
import pyfim

order_data = pd.read_csv("./order_products__train.csv")
product_data = pd.read_csv("./products.csv")
named_orders = pd.merge(order_data, product_data, on="product_id")
counts = named_orders["product_name"].value_counts()
counts = counts[counts > 1000]

selected_orders = named_orders[named_orders["product_name"].isin(counts.index.values.tolist())]
selected_orders["cols"] = selected_orders.groupby("order_id").cumcount()
selected_pivot = selected_orders.pivot(index = "order_id", columns = "cols")[["product_name"]]
selected_pivot.head()

purchases = []
for i in range(0,len(selected_pivot)):
    purchases.append([str(selected_pivot.values[i,j]) for j in range(0,25)])

rules = apriori(purchases, min_support = .01, min_confidence=0.1, min_lift = 3, min_length=20 )
results = list(rules)

results[0]
#1.6% purchased a Lime
#19.6% of the time if someone bought a Lime, they also bought a Large Lemon
#3.2% Improvement in confidence given we bought a Lime

rule_counter = 0
for i in range(0, len(results)):
    result = results[i]
    supp = int(result.support*10000)/100
    conf = int(result.ordered_statistics[0].confidence*100)
    hypo = ''.join([x+' ' for x in result.ordered_statistics[0].items_base])
    conc = ''.join([x+' ' for x in result.ordered_statistics[0].items_add])
    if "nan" not in hypo:
        rule_counter = rule_counter + 1
        print("If "+str(hypo)+ " is purchased, "+str(conf)+" % of the time "+str(conc)+" is purchased [support = "+str(supp)+"%]")
print("Total rules built, omitting NaN: "+str(rule_counter))

#Ecalt
rules = pyfim.eclat(purchases, supp=2, zmin=2, out=[])

rule_count=0
for i in range(0, len(rules)):
    supp = round(int(rules[i][1]) / len(purchases)*100,3)
    items = rules[i][0]
    if "nan" not in items:
        rule_count = rule_count + 1
        item_1 = rules[i][0][0]
        item_2 = rules[i][0][1]
        print("If "+str(item_1)+ " is purchased, "+str(supp)+" % of the time "+str(item_2)+" is purchased [absolute support = "+str(supp)+"%]")
print("Total rules built, ommitting NaN: " +str(rule_count))
        
#UCB
from slots import playSlots

cash = 1000
spin = 1
while(cash > 1):
    print("Spin: " + str(spin))
    winnings = playSlots(machine=10)
    print("I won! I got "+ str(winnings))
    cash = (cash - 1) + winnings
    print("I now have $" + str(cash))
    spin = spin+1

import random
N = 2000
d = 20
machine_record = []
award_record = []
balance = 1000.0
for n in range(0,N):
    machine_choice = random.randrange(1,d+1)
    machine_record.append(machine_choice)
    cash_reward = playSlots(machine = machine_choice)
    award_record.append(cash_reward)
    balance = (balance - 1.0) + cash_reward

plt.hist(machine_record, bins=20)
plt.title("Machine Selection (End Balance: "+str(round(balance,2)) + ")")
plt.xlabel("Slot Machine")
plt.ylabel("Number of Times Used")
plt.show()

import random
import math
N = 2000
d = 20
machine_record = []
award_record = []
balance = 1000.0

number_of_selections = [0] * d
sum_of_rewards = [0] * d

for n in range(0,N):
    machine_choice=1
    max_upper_bound=0
    
    for i in range(1,d):
        if(number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2* math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e200
            
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            machine_choice = i
    
    machine_record.append(machine_choice)
    number_of_selections[machine_choice] = number_of_selections[machine_choice] + 1
    award = playSlots(machine = machine_choice)
    sum_of_rewards[machine_choice] = sum_of_rewards[machine_choice] + award
    balance = balance - 1 + award
    
plt.hist(machine_record, bins=20)
plt.title("Machine Selection (End Balance: "+str(round(balance,2)) + ")")
plt.xlabel("Slot Machine")
plt.ylabel("Number of Times Used")
plt.show()

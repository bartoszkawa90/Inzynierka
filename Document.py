####  Example of multiple threads in Python
def count(name):
    for i in range(1000):
        print(str(i) + name + "\n")

import threading


##  multiple threads in a loop
names = ["thread_1", "thread_2", "thread_3"]

for name in names:
    thread = threading.Thread(target=count, args=(name, ))
    thread.start()

##  Single threads one after another
# name1 = "thread_1"
# thread_1 = threading.Thread(target=count, args=(name1,))
# thread_1.start()
#
# name2 = "thread_2"
# thread_2 = threading.Thread(target=count, args=(name2, ))
# thread_2.start()
#
# name3 = "thread_3"
# thread_3 = threading.Thread(target=count, args=(name3, ))
# thread_3.start()

#-----------------------------------------------------------------------------------------------------------------------



#!/usr/bin/python
import sys
from pyspark import SparkConf, SparkContext

'''
    Let v be a given vertex
    gamma(v) is then defined as the neighbouhood of v i.e.
        it is the set of all the vertices connected to v
    gamma_plus(v) is then gamma(v) + {v}
'''

'''
    Connects all the strictly large neighbours in gamma(v)
    to min(gamma_plus(v))
'''


def large_star(x):
    global changes
    v, gamma_plus = x[0], x[1]

    # Add the vertex to actually make it gamma_plus
    gamma_plus.append(v)

    # Get the minimum vertex
    min_vertex = min(gamma_plus)



    # Emit
    # Gamma and gamma plus doesn't affect as when x=v x==v
    to_return = []
    for x in gamma_plus:
        if x>v:
            to_return.append((x, min_vertex))
            # If v is not the min_vertex and larger neighbours present
            if min_vertex != v:
                changes += 1
    return to_return


'''
    Connect all the smaller neighbours including self
    to min(gamma(v))
'''


def small_star(x):
    global changes
    v, gamma = x[0], x[1]
    min_vertex = min(gamma)

    # Make gamma to gamma_plus by appending v
    gamma.append(v)

    # Connect all the smaller neighbours including self to min(gamma(v))
    # don't connect min vertex to itself
    to_return = []
    for x in gamma:
        if x <= v and x != min_vertex:
            to_return.append((x, min_vertex))
            # If v is not the minimum and has a lesser vertexes to connected to minimum
            if x < v:
                changes += 1
    return to_return


if __name__ == "__main__":
    if(len(sys.argv)!=2):
        print('Usage python a2.py <graph_in_file>')
        sys.exit(0)

    
    file_name = sys.argv[1]
    save_as_file = 'graph_output'
    delimitter = ' '
    
    conf = SparkConf().setAppName("connected_components")
    sc = SparkContext(conf=conf)

    # Load the data file
    lines = sc.textFile(file_name)

    # To keep a track of the changes
    changes = sc.accumulator(0)


    def parse_txt(line): return [int(x) for x in line.split(delimitter)]


    def large_star_map(x): return [(x[0], [x[1]]), (x[1], [x[0]])]


    def small_star_map(x): return (
        x[0], [x[1]]) if x[1] <= x[0] else (x[1], [x[0]])


    def list_extend(x, y): return x+y


    # Parse the read edges
    edges = lines.map(parse_txt)
    out = edges

    prv_val = 0

    i=0
    # print('-'*70)
    while True:

        l_star = out.flatMap(large_star_map).reduceByKey(
            list_extend).flatMap(large_star)

        s_star = l_star.map(small_star_map).reduceByKey(
            list_extend).flatMap(small_star)


        # Force spark to evaluate the data -> use take(1) for minimal communication
        _ = s_star.take(1)
        # print('total changes in the iteration are',changes.value - prv_val)

        #If no changes then break!!!
        if changes.value - prv_val == 0:
            break

        # Update the iterators
        prv_val = changes.value
        out = s_star
        i+=1

        ##Another convergence condition -> slower
        #if l_star.subtract(s_star).union(s_star.subtract(l_star)).isEmpty():
        #    break


    # Add self loop for indicating vertices belonging to the class
    self_nodes = out.values().distinct().map(lambda x: (x, x))
    combined = out.union(self_nodes)
    # print(combined.values().distinct().collect())
    combined.saveAsTextFile(save_as_file)

    sc.stop()

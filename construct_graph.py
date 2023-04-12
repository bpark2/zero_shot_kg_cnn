# graph structure preparation file 
import json
import requests
import logging
import time

# set-up the parameters 
BFS_DEPTH = 3
TARGET_INCREMENT = 100
CHUNK_DELAY_SECONDS = 1
NODE_DELAY_SECONDS = 3
FAIL_DELAY_SECONDS = 2
NODE_START = 0 # specify the starting place in the wikidata mapping 
NODE_END = 5 # specify the end node 
OUTPUT_FILE_NAME = f"data/graph_struct/{str(BFS_DEPTH)}-hops_graph_{str(NODE_START)}-{str(NODE_END)}.json"


# create a logger, because ... things could go wrong and this runs a long time
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'data/graph_struct/construct_graph_{BFS_DEPTH}_{NODE_START}_{NODE_END}.log')
fh.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# building the query
def build_target_string(target_ids): 
  """ Create the string specifying all the targets. For use in build_query.

  Args: 
    target_ids: list of Wikidata ID strings for all targets
  """
  string_list = ["gas:program gas:target " + x + " . " for x in target_ids]
  return "\n".join(string_list)

def build_query(input_id, targets, k): 
  """ Assemble the query using the template. 

  Args: 
    input_id: the Wikidata ID of the root of the search tree
    targets: the string representing the list of targets
    k: the depth of the BFS
  """
  out = ('PREFIX gas: <http://www.bigdata.com/rdf/gas#> \n'
  'SELECT ?depth ?predecessor ?linkType ?out { \n'
  'SERVICE gas:service { \n'
     'gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.BFS" . \n'
     'gas:program gas:in ')
  out += input_id + ' . \n'
  out += targets + "\n"
  out += ('gas:program gas:out ?out . \n'
     'gas:program gas:out1 ?depth . \n'
     'gas:program gas:out2 ?predecessor . \n'
     'gas:program gas:maxIterations ')
  out += str(k) + "\n"
  out += (  '} \n'
    '?predecessor ?linkType ?out . \n'
    '} \n' 
    'order by desc(?depth)\n'
    'limit 1000 \n')
  return out

# actually make the request of the SPARQL endpoint 
# code adapted from: 
def wikidata_query(query):
    url = 'https://query.wikidata.org/sparql'
    try:
        r = requests.get(url, params = {'format': 'json', 'query': query})
        return r.json()['results']['bindings']
    except json.JSONDecodeError as e:
       raise 

# methods for processing the results of the query
def process_row(row): 
  row['out'] = row['out']['value'].split("/")[-1]
  row['predecessor'] = row['predecessor']['value'].split("/")[-1]
  row['depth'] = row['depth']['value']
  row['linkType'] = row['linkType']['value']

def update_adj_dict(row): 
  global adj_dict
  node_A = row['predecessor']
  node_B = row['out']
  if node_A not in adj_dict.keys():
    adj_dict[node_A] = [node_B]
  else: 
    adj_dict[node_A].append(node_B)
  
  if node_B not in adj_dict.keys():
    adj_dict[node_B] = [node_A]
  else: 
    adj_dict[node_B].append(node_A)

def process_query_result(result): 
  for row in result: 
    process_row(row)
    if row['depth'] == str(0): 
      continue # it's an incoming link to the source node 
    else: 
      update_adj_dict(row)

def process_chunk(source, targets, k = 5, max_tries = 2): 
  """ Attempts to query and process results for one "chunk" of target nodes

  Args: 
    source: the Wikidata ID string for the source node
    targets: the list of Wikidata IDs to list as target nodes
    k: (optional) the depth to run the BFS at 
    max_tries: the number of times to attempt the query (in the case of an exception, likely a timeout error)

  Returns: 
    True if the request and processing succeeded, False if the query failed. 
  """

  # build query 
  target_str = build_target_string(targets)
  query = build_query(source, target_str, k)

  # try the query
  for i in range(0, max_tries):
    try: 
      result = wikidata_query(query)
    except Exception as e: 
      logger.warning(f"Query for {source} failed. Likely due to timeout. Trying again...") 
      if i == max_tries-1: 
        logger.warning(f"Max tries exceeded for query.")
        raise
    else: 
      break
  
  # process query results 
  process_query_result(result)

  return True

def process_node(source_node, targets, k = 5, increment = 25): 
  i = increment
  n = len(targets)
  while i < n:
    end = min(i+1, n)
    try:
      process_chunk(source_node, targets[(i-increment) : end], k)
      time.sleep(CHUNK_DELAY_SECONDS)
    except Exception as e: 
      logger.error(f"Processing failed on node {source_node} for chunk {i-increment}:{end} at k = {k}\n\n" + str(e))
      time.sleep(FAIL_DELAY_SECONDS)
    i += increment
  logger.info(f"Node {source_node} processed.")


### MAIN EXECUTION ###

# read the list of entities 
with open("data/graph_struct/wikidata_ids.json") as id_file: 
    nodes = json.load(id_file)

# subset for the source nodes to run, need to keep nodes intact for the target nodes
source_nodes = nodes[NODE_START : NODE_END]
n = len(source_nodes)
adj_dict = dict()

logger.info("Beginning graph construction...")
with open(OUTPUT_FILE_NAME, "w") as out: 
    out.write("[") 
    i = 0

    for i in range(0, n): 
        # clear out the dictionary
        adj_dict = dict()
        process_node(source_nodes[i], nodes, k = BFS_DEPTH, increment = TARGET_INCREMENT)
        out.write(json.dumps(adj_dict))
        i += 1
        logger.info(f"{i} / {n} nodes processed.")
        time.sleep(NODE_DELAY_SECONDS)
        if i != n: 
          out.write(", ")

    out.write("]")

logger.info("Processing complete.")
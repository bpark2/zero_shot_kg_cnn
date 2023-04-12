import json 
# this script creates the list of ids for the image nodes in the KG

OUTPUT_FILENAME = "data/graph_struct/wikidata_ids.json"



# create the list of Wikidata IDs to include in the graph 
def get_Wikidata_ID_list(): 
  """ Create a list of all the Wikidata IDs for the ImageNet classes and the snake classes

  Returns:
    list of strings for all Wikidata IDs that need to be included in the graph (i.e. we have images for them)
    all ids are prepended with "wd:" for immediate use in SPARQL queries at the Wikidata endpoint
  """
  with open('data/graph_struct/mapping.json') as map_file: 
    mapping = json.load(map_file)
  wd_urls = []
  for key in mapping:
    wd_urls.append(mapping[key])

  wd_ents = [x.split("/")[-1] for x in wd_urls]
  snakes = ['Q2065834', 'Q215388', 'Q43373908', 'Q1248772', 'Q2998807', 'Q945692', 
            'Q2998811', 'Q905354', 'Q2163958', 'Q2248178', 'Q426062', 'Q2565456']
  wd_ents.extend(snakes)
  wd_ents = list(set(wd_ents))
  wd_ents = ["wd:" + id for id in wd_ents] # append the wd: tag for use in Wikidata queries
  return wd_ents

# create the list 
ents = get_Wikidata_ID_list()

# save to file
with open(OUTPUT_FILENAME, "w") as out: 
  json.dump(ents, out)
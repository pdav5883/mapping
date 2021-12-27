# mapping

Visualizing different map projections with simple vector data.

Dataset is country borders geojson from Natural Earth, which provides vector data in lat/lon degrees. `CreateMapData` notebook converts the geojson data into numpy arrays and saves them in binary format for easier loading.

File `map_proj.py` contains all of the projection functions, which mostly follow the same conventions of map center/direction, as well as map plotting functions. `MappingScratch` notebook contains cells that visualize each of the different projections. Note that more efficient/robust python modules exist to perform these same projections, but the intent of this repo was to go through the projection math myself.

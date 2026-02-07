# Wildbirds Layer
All code is in the ``./scripts`` folder.

``convert_weekly_birds.py``: This script converts weekly_birds.zip contents
to a csv file with the glw and fips values. This is used subsequently in
the portal as well as risk assessment.

``process_birds.py``: This script processes the weekly species-specific
abundance data to generate total abundance data for each grid cell.

The remaining scripts are used for aligning eBirds geometry with GLW geometry.
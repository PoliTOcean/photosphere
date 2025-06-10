Stitched equator + poles version: 
0. In the "photos" folder, two images are expected to be named north.jpg and south.jpg. These will be mapped to the photosphere poles.
1. after the "photos" folder has been properly filled, run "start stitched.bat". You should now see a new window with the rendered photosphere.
2. By pressing Y or X and using the mouse wheel, you can dynamically change the mapping of the equator on the two axes. By pressing P you can change the size of the poles.
3. Using + and - buttons, you can change the latitude at which the poles and the equator meet, improving visibility case to case.
4. Q/W and E/R to rotate the poles, respectively CW and CCW. 

Fixed sectors + poles version: 
1. Open the "sphere render.py" file and modify the NUM_EQUATOR_SECTORS parameter to change the number of sectors the photosphere will be divided. In the "photos" folder, at least NUM_EQUATOR_SECTORS images are expected, named from "sector_1.jpg" up to "sector_[NUM_EQUATOR_SECTORS].jpg", plus the north.jpg and the south.jpg images.
2. after the "photos" folder has been properly filled, run "sphere render.py". You should now see a new window with the rendered photosphere.
3. By using the mouse wheel, you can dynamically change the size of sectors and poles images.
4. Using + and - buttons, you can change the latitude at which the poles and the sectors meet, improving visibility case to case.
5. Q/W and E/R to rotate the poles, respectively CW and CCW. 

In both cases use mouse left to rotate the 3D view.

A toggable debug is present, that logs user changes to the crop parameters and to the poles max latitudes and rotations.
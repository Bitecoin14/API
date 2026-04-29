HOW TO ADD A NEW PERSON
=======================

1. Create a sub-folder inside known_faces/ using the person's name.
   Examples:
     known_faces/Alice/
     known_faces/Bob_Smith/

2. Copy one or more clear, front-facing photos of that person into the folder.
   Supported formats: .jpg  .jpeg  .png

3. Restart hand_tracker.py. The program loads encodings at startup.

TIPS
----
- More photos (3-10) -> better accuracy.
- Photos should show the face clearly, without heavy shadows or extreme angles.
- File names inside the folder don't matter.
- To remove a person, delete their folder and restart.

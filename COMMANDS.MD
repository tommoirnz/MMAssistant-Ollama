This document details all voice/text commands recognized by the AI assistant system. Commands bypass the AI and trigger immediate actions when detected in user input.

🔌 SYSTEM STATUS COMMANDS
💤 Sleep Mode
Command	Action	Notes
"sleep"	Enter sleep mode	Stops processing voice input
"go to sleep"	Enter sleep mode	-
"rest mode"	Enter sleep mode	-
"silence"	Enter sleep mode	-
"stop listening"	Enter sleep mode	-

🔔 Wake Mode (only works when sleeping)
Command	Action	Notes
"wake"	Exit sleep mode	Resumes normal listening
"wake up"	Exit sleep mode	-
"awaken"	Exit sleep mode	-
"resume"	Exit sleep mode	-
"start listening"	Exit sleep mode	-
"listen again"	Exit sleep mode	-

🎤 AUDIO CONTROL
⏹️ Stop Speaking
Command	Action	Notes
"stop"	Immediately stop TTS	Halts any ongoing speech
"stop speaking"	Immediately stop TTS	-
"stop talking"	Immediately stop TTS	-
"be quiet"	Immediately stop TTS	-
"shut up"	Immediately stop TTS	-
"enough"	Immediately stop TTS	-
"that's enough"	Immediately stop TTS	-
"okay stop"	Immediately stop TTS	-
"ok stop"	Immediately stop TTS	-


🔇 Mute Control (Single AI)
Command	Action	Notes
"mute"	Toggle AI audio	Mutes/unmutes TTS output
"mute audio"	Toggle AI audio	-
"silence"	Toggle AI audio	-
"mute the ai"	Toggle AI audio	-
"unmute"	Toggle AI audio	-
"enable audio"	Toggle AI audio	-
"audio on"	Toggle AI audio	-
"toggle mute"	Toggle AI audio	-
"mute toggle"	Toggle AI audio	-

📸 VISION & CAMERA COMMANDS
📷 Camera Control
Command	Action	Notes
"start camera"	Open/start camera	Opens Image Window
"load camera"	Open/start camera	-
"turn on camera"	Open/start camera	-
"camera on"	Open/start camera	-
"start the camera"	Open/start camera	-
"enable camera"	Open/start camera	-
"stop camera"	Stop camera feed	-
"close camera"	Stop camera feed	-
"turn off camera"	Stop camera feed	-
"camera off"	Stop camera feed	-
"stop the camera"	Stop camera feed	-
"disable camera"	Stop camera feed	-
"take a picture"	Capture snapshot	Saves to out/snapshot_*.png
"take picture"	Capture snapshot	-
"take a photo"	Capture snapshot	-
"take photo"	Capture snapshot	-
"snapshot"	Capture snapshot	-
"capture photo"	Capture snapshot	-
"capture image"	Capture snapshot	-
"snap a picture"	Capture snapshot	-

🖼️ Image Analysis Commands
Static Images (analyzes uploaded files)
Command	Action	Notes
"describe the image"	Analyze current image	Uses last uploaded image
"describe what's in the image"	Analyze current image	-
"describe what is in the image"	Analyze current image	-
"what's in the image"	Analyze current image	-
"what is in the image"	Analyze current image	-
"analyze the image"	Analyze current image	-
"explain the image"	Analyze current image	-
"what does the image show"	Analyze current image	-
Live Camera View (triggers camera + snapshot)

Command	Action	Notes
"what do you see"	Live camera analysis	Opens camera, takes snapshot
"what can you see"	Live camera analysis	-
"whats happening"	Live camera analysis	-
"what is happening"	Live camera analysis	-
"whats going on"	Live camera analysis	-
"what is going on"	Live camera analysis	-
"describe what you see"	Live camera analysis	-
"tell me what you see"	Live camera analysis	-
"show me what you see"	Live camera analysis	-
"what are you seeing"	Live camera analysis	-
"whats in front of you"	Live camera analysis	-
"describe the scene"	Live camera analysis	-
"whats around you"	Live camera analysis	-
"whats in the room"	Live camera analysis	-


👥 PERSONALITY SWITCHING
Personality Activation Commands
Command	Personality	Description
"be the butler"	Jeeves the Butler	Formal, helpful butler
"butler mode"	Jeeves the Butler	-
"butler personality"	Jeeves the Butler	-
"be the teacher"	Mrs. Hardcastle	Strict school teacher
"teacher mode"	Mrs. Hardcastle	-
"school teacher"	Mrs. Hardcastle	-
"strict teacher"	Mrs. Hardcastle	-
"be the scientist"	Dr. Von Knowledge	Mad scientist
"mad scientist"	Dr. Von Knowledge	-
"scientist mode"	Dr. Von Knowledge	-
"dr von knowledge"	Dr. Von Knowledge	-
"be default"	Default	Reset to default
"be normal"	Default	-
"normal mode"	Default	-
"be the explorer"	Space Explorer	Space explorer persona
"space explorer mode"	Space Explorer	-
"space explorer personality"	Space Explorer	-
"captain nova mode"	Space Explorer	-
"om shanti"	BK Yogi	Spiritual/meditative
"aum shanti"	BK Yogi	-
"be the computer"	Holly	Red Dwarf ship computer
"be holly"	Holly	-
"be the ship computer"	Holly	-
"holly mode"	Holly	-
"red dwarf mode"	Holly	-
"computer personality"	Holly	-
"ship computer"	Holly	-
"talk to holly"	Holly	-
"wake up holly"	Holly	-
"be rab"	Rab C. Nesbitt	Glaswegian philosopher
"rab c nesbitt"	Rab C. Nesbitt	-
"glasgow mode"	Rab C. Nesbitt	-
"scottish philosopher"	Rab C. Nesbitt	-
"be the glaswegian"	Rab C. Nesbitt	-
"rab mode"	Rab C. Nesbitt	-
"talk to rab"	Rab C. Nesbitt	-
"glasgow drunk"	Rab C. Nesbitt	-
"street philosopher"	Rab C. Nesbitt	-
"nesbitt mode"	Rab C. Nesbitt	-

📊 PLOTTING COMMANDS
🔧 Plotting Control
Enabled by: "Plotting ON" checkbox in UI (default: ✓)

Disabled by: "Plotting OFF" checkbox in UI

📈 Direct Plot Commands
Command	Action	Notes
"plot"	Graph last expression	Looks in LaTeX window
"graph"	Graph last expression	-
"draw"	Graph last expression	-
"show the graph"	Graph last expression	-
"plot the function"	Graph last expression	-
"graph that"	Graph last expression	Most common
"plot that"	Graph last expression	-
"plot the result"	Graph last expression	-
"plot the equation"	Graph last expression	-
"draw the graph"	Graph last expression	-
"show me a plot"	Graph last expression	-
"show a plot"	Graph last expression	-
"display the graph"	Graph last expression	-
"visualize the function"	Graph last expression	-
"create a plot"	Graph last expression	-
"make a graph"	Graph last expression	-
"draw a plot"	Graph last expression	-
"graph the function"	Graph last expression	-
"plot the graph"	Graph last expression	-
"graph the result"	Graph last expression	-
"draw the function"	Graph last expression	-
"plot that function"	Graph last expression	-
"graph that equation"	Graph last expression	-
"show me the graph"	Graph last expression	-
"make a plot"	Graph last expression	-
"create a graph"	Graph last expression	-
"visualize"	Graph last expression	-
"show graphically"	Graph last expression	-
"plot it"	Graph last expression	-
"graph it"	Graph last expression	-
"draw it"	Graph last expression	-
"plot this function"	Graph last expression	-
"graph this equation"	Graph last expression	-
"show as a graph"	Graph last expression	-
"draw the equation"	Graph last expression	-
"plot the expression"	Graph last expression	-
"graph the expression"	Graph last expression	-


🔍 WEB SEARCH COMMANDS
Command	Action	Notes
"search for"	Web search	Opens search window
"search"	Web search	-
"look up"	Web search	-
"information on"	Web search	-
"find information about"	Web search	-
"web search"	Web search	-
"search the web for"	Web search	-
"look for"	Web search	-
"that's for"	Web search	-
"database"	Web search	-
"online"	Web search	-
Behavior:

Opens Web Search Window

Uses Brave Search API

Summarizes results with AI

Speaks summaries with TTS

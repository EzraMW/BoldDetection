@echo off
:start
cls


This is all the code to process the image into the xml form abbyy gives

cd C:\Users\emwil\Downloads\bad_pics\low score images


for %%a in (*) do (
	echo pic: C:\Users\emwil\Downloads\bad_pics\low score images\%%a
	echo xml: C:\Users\emwil\Downloads\bad_xml\%%a.xml
	echo next xml
	python "C:\Users\emwil\cs_projects\Structure Detection\process.py" -l "Hebrew" -xml "C:\Users\emwil\Downloads\bad_pics\low score images\%%a" "C:\Users\emwil\Downloads\bad_xml\%%a.xml"
)

pause
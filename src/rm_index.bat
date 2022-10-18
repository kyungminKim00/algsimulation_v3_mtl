@echo off
rem set root_dir=%cd%
rem set /p str=Data Version:
rem set split=%str%
rem for /f %%f in ('dir /B %split%') do (
    rem echo del /S %split%\%%f\fig_index\index
	rem echo del /S %split%\%%f\validation\fig_index\index
	rem del /S /Q %split%\%%f\fig_index\index
	rem del /S /Q %split%\%%f\validation\fig_index\index
rem )

set list=v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61
for %%a in (%list%) do (
	for /f %%f in ('dir /B %%a') do (
		echo del /S /Q %%a\%%f\fig_index\index
		echo del /S /Q %%a\%%f\validation\fig_index\index
		del /S /Q %%a\%%f\fig_index\index
		del /S /Q %%a\%%f\validation\fig_index\index
))

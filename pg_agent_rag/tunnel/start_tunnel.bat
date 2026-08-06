@echo off
REM Usage:
REM   start_tunnel.bat connect.nmb1.seetacloud.com 46572
REM Maps local 18180 -> AutoDL 6006

set HOST=%1
set PORT=%2
if "%HOST%"=="" (
  echo Usage: start_tunnel.bat ^<autodl-host^> ^<ssh-port^>
  echo Example: start_tunnel.bat connect.nmb1.seetacloud.com 46572
  exit /b 1
)
if "%PORT%"=="" (
  echo Missing SSH port
  exit /b 1
)

echo Tunnel: localhost:18180 -^> root@%HOST%:%PORT% -^> 127.0.0.1:6006
echo Test after connect: curl http://127.0.0.1:18180/health
ssh -CNg -L 18180:127.0.0.1:6006 root@%HOST% -p %PORT%

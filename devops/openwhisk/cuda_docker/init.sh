cd /action;
mkfifo /action/fifo.1;
chmod 777 /action/fifo.1;

while [ true ]
do
	if [ -f /action/exec.exe ]
	then
		cd /action;
		chmod +x /action/exec.exe;
	
		if [ -p /action/fifo.1 ] 
		then			
			while [ true ]
			do
				/bin/bash -c "/action/exec.exe 2>>log &"
				exe_pid=$(pidof exec.exe)
				if [[ -n "$exe_pid" ]]
				then
					break;
				fi
			done
		fi
		break;
	fi
done

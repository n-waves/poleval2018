die() { echo "$*" 1>&2 ; exit 1; }

force_bool() { 
    if [ "$2" != "True" ] && [ "$2" != "False" ];
        then die "$1 must be 'True' of 'False', but '$2' was given"
    fi
}


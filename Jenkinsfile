pipeline {
    agent {
        docker {
            image 'amiller27/iarc7-base'
            args '-u root'
        }
    }
    stages {
        stage ('Setup-Workspace') {
            steps {
                sh '''
                    set -x
                    cd ~
                    mkdir -p catkin_ws/src
                    cd catkin_ws/src
                    source /opt/ros/kinetic/setup.bash
                    catkin_init_workspace
                    cd ..
                    catkin_make
                    source devel/setup.bash
                    cd src
                    ln -s $WORKSPACE jenkins_test_package
                    '''
            }
        }
        stage ('Setup-Dependencies') {
            steps {
                sh '''
                    set -x
                    cd ~/catkin_ws/src
                    wstool init
                    if [[ -f "$WORKSPACE/dependencies.rosinstall" ]]; then
                        wstool merge "$WORKSPACE/dependencies.rosinstall"
                    fi
                    wstool update
                    cd ..
                    rosdep install -y --from-paths src --ignore-src --rosdistro kinetic
                    '''
            }
        }
        stage ('Build') {
            steps {
                sh '''
                    set -x
                    source /opt/ros/kinetic/setup.bash
                    cd ~/catkin_ws
                    catkin_make
                    '''
            }
        }
        stage ('Test') {
            steps {
                sh '''
                    set -x
                    cd ~/catkin_ws
                    source devel/setup.bash
                    catkin_make run_tests && catkin_make test
                    '''
            }
        }
    }
}

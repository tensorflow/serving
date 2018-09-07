#!groovy
pipeline {
	agent any
		environment {
			def gitCommit = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
			def gitUrl = sh(returnStdout: true, script: 'git config remote.origin.url').trim()
		}
	stages {
		stage('Build Docker images'){
			environment {
				QUAY_USERNAME = credentials('QUAY_USERNAME')
				QUAY_PASSWORD = credentials('QUAY_PASSWORD')
			}
			steps {
				sh 'QUAY_USERNAME=${QUAY_USERNAME} QUAY_PASSWORD=${QUAY_PASSWORD} GIT_URL=${gitUrl} GIT_BRANCH=${BRANCH_NAME} GIT_COMMIT=${gitCommit} make build'
			}
		}
		//stage('Deploy') {
			//steps {
				//sh 'GIT_URL=${gitUrl} GIT_BRANCH=${BRANCH_NAME} GIT_COMMIT=${gitCommit} /usr/local/bin/kairos-deploy-helm.sh'
			//}
		//}
		//stage('Run Runscope Tests') {
			//steps {
				//sh 'GIT_BRANCH=${BRANCH_NAME} scripts/test-runner.sh'
			//}
		//}
		//stage('Approval') {
			//when {
				//branch 'master'
			//}
			//steps {
				//timeout(time:5, unit:'DAYS') {
					//input message: 'Deploy to Production?'
				//}
			//}
		//}
		//stage('Deploy to Prod') {
			//when {
				//branch 'master'
			//}
			//steps {
				//sh 'GIT_URL=${gitUrl} GIT_BRANCH=${BRANCH_NAME} GIT_COMMIT=${gitCommit} /usr/local/bin/kairos-deploy-helm.sh prod'
			//}
		//}
	}

	post {
		success {
			slackSend  failOnError: true,
				channel: '#jenkins',
				color: '#139C8A',
				message: "BUILD SUCCESS:\n  JOB: ${env.JOB_NAME} [${env.BUILD_NUMBER}]\n  BUILD URL: ${env.BUILD_URL}"
		}

		failure {
			slackSend  failOnError: true,
				channel: '#jenkins',
				color: '#FF6347',
				message: "BUILD FAILURE:\n  JOB: ${env.JOB_NAME} [${env.BUILD_NUMBER}]\n  BUILD URL: ${env.BUILD_URL}"
		}

		unstable {
			slackSend  failOnError: true,
				channel: '#jenkins',
				color: '#1175E3',
				message: "BUILD UNSTABLE:\n  JOB: ${env.JOB_NAME} [${env.BUILD_NUMBER}]\n  BUILD URL: ${env.BUILD_URL}"
		}

		always {
			// Recursively delete all files and folders in the workspace
			//sh 'make clean'
			deleteDir()
		}
	}

	options {
		buildDiscarder(logRotator(numToKeepStr:'10'))
		timeout(time: 60, unit: 'MINUTES')
	}
}

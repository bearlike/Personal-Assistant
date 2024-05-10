#!/usr/bin/env python3
# TODO: Complete the API submodule by wrapping around the Meeseeks core.
import os
import sys
from flask import Flask, request
from flask_restx import Api, Resource
# TODO: Need to package the application and import it as module
# Adding the parent directory to the path before importing the custom modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Custom imports - Meeseeks core modules
from core.task_master import generate_action_plan, run_action_plan

app = Flask(__name__)
api = Api(app)


@api.route('/generate_action_plan')
class GenerateActionPlanResource(Resource):
    def post(self):
        user_input = request.json.get('user_input')
        action_plan_list, task_queue = generate_action_plan(user_input)
        return {'action_plan_list': action_plan_list, 'task_queue': task_queue}


@api.route('/run_action_plan')
class RunActionPlanResource(Resource):
    def post(self):
        task_queue = request.json.get('task_queue')
        ai_response = run_action_plan(task_queue)
        return {'ai_response': ai_response}


if __name__ == '__main__':
    app.run(debug=True)

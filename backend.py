#!/usr/bin/env python3
from flask import Flask, request
from flask_restx import Api, Resource
from chat_master import generate_action_plan_helper, run_action_plan_helper

app = Flask(__name__)
api = Api(app)


@api.route('/generate_action_plan')
class GenerateActionPlanResource(Resource):
    def post(self):
        user_input = request.json.get('user_input')
        action_plan_list, task_queue = generate_action_plan_helper(user_input)
        return {'action_plan_list': action_plan_list}


@api.route('/run_action_plan')
class RunActionPlanResource(Resource):
    def post(self):
        task_queue = request.json.get('task_queue')
        ai_response = run_action_plan_helper(task_queue)
        return {'ai_response': ai_response}


if __name__ == '__main__':
    app.run(debug=True)

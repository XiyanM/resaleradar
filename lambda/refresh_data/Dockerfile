FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda_function.py features.py retrain.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.handler"]

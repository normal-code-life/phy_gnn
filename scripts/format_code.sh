#!/usr/bin/env bash

echo "-- black --"
black --line-length=120 .

echo "-- isort --"
isort .

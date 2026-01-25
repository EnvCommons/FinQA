from openreward.environments import Server

from finqa import FinQA

if __name__ == "__main__":
    server = Server([FinQA])
    server.run()

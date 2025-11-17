from sales_management_dashboard import app

# Required by Vercel
server = app.server

def handler(event, context):
    return server(event, context)

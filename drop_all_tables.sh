#!/bin/bash

# MySQL credentials
user="CBaSE"
password="darwin1809"
db="CBaSE"

# Create a variable with the command to list all tables
tables=$(mysql -u $user -p$password -Nse 'SHOW TABLES' $db)

# Loop through the tables and drop each one
for table in $tables; do
    echo "Dropping $table from $db..."
    #echo "mysql -u $user -p $password -e "DROP TABLE \`$table\`" $db;"
    mysql -u $user -p$password -e "DROP TABLE \`$table\`" $db;
done

echo "All tables dropped from $db."
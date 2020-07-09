
Elaborating marketing strategy for a mythical insurance company via unsupervised machine learning.
Data description and conclusions:
 
Here follows the description of variables, and its initially assumed value from a business perspective:
ID - possesses no valuable information for decision-making, therefore not used for further analysis.

FIRST POLICY YEAR represents customer loyalty to the company. BIRTHDAY YEAR is assumed to be correlated with monthly income. 

FIRST POLICY YEAR and BIRTHDAY YEAR both have numeric values and also it was detected that either of the columns have inaccurate data.In 2204 records FIRST POLICY YEAR exceeds BIRTHDAY YEAR which is not logical. This issue is addressed further in the data-preprocessing section.

EDUCATION is a nominal variable and it is certainly assumed to be highly correlated with gross monthly salary.

GEOGRAPHICAL LIVING AREA is a nominal variable, even though it has integers as values, it is not considered a numeric attribute because the integers are not meant to be used quantitatively. It is initially assumed that living area might be correlated with customers’ income or can affect the choice of certain products, for example, people living in suburbs might be more frequent in buying motor insurance.

CHILDREN is a binary variable and represents the parenthood status of the client. Can be correlated with investments in Life in Health insurances.

CUSTOMER MONETARY VALUE, CLAIMS RATE and all kinds of PREMIUMS are numeric variables that are indicators of clients’ profit-making capacity for the company.

PREMIUMS are numeric variables representing the amount spent in a certain insurance product.

CONCLUSIONS and SUGGESTED MARKETING CAMPAIGN

1. "Customer Value" Clusters

It is obvious from the results of K-means clustering that there are 4 obvious and easily interpretable clusters.
The first cluster identified is “Ideal” customers with the following characteristics: high average salary and low claim rate - these people are company’s best customers since they generate income and could be exploited further, for instance new insurance services, if absent, could be offered. These customers are expected to keep claim rate low when using other products as well, since due to their high income, they, probably, do not tend to submit their claims once a minor accident arises.
Second cluster is “Potentially Ideal” customers and have high average salary and high claim rate. These customers are less valuable from profit-making point of view, however, have resources to use other insurance products in which they would probably have less insurance events.
Third cluster is “Stable” or “Good Enough” customers and have they low average salary and low claim rate. Certainly, from a business point of view, low claim rate is highly appreciated, and even though, these customers would not be able to use more services, they are highly valuable for the business.
Forth cluster is “Bad” customers’ cluster because they have low average salary and high claim rate and therefore not considered to be an attractive investment.

2. "Consumption" Clusters

Based on the result of K-Means there are 3 clusters resulting from Consumption dataset. There are “health, motor, household”, “household” and “motor” clusters  according to the K-means - the names of the clusters are based on their higher spending.
The cluster have the following characteristics:
All the clusters have very little relative value of consumption of "Life" and "Work Compensations"- on average around 0.5-0.7% from the total consumption.
“Health, motor and household” cluster seems to have the most equal distribution of other three products consumed compared to other clusters since it has on average 35%, 30% and 35% of spendings in health, household and motor respectively.
“Household” cluster has greater spendings in insurance of the household (on average 81%) and spendings in “Motor” cluster are dominated by spendings in car insurance (on average 65%)

The cross-tables of the Consumption Clusters show that, on average, “Motor" cluster clients have higher education and more present parenthood.
While the “Household” cluster clients seem to be the less educated and with the presence of children between the other two clusters. 
Last, the “Health, motor, household” cluster shows medium education clients and the least presence of children out of all the clusters.
We could infer from these data that the “Motor” cluster has clients with a higher average age than the others, since it seems to have more PhD clients and 
a higher presence of children.
These tables also could help us define our marketing strategies.For instance, we know that “Bad” Clients cluster and the “Motor” cluster both have high 
presence of children,which means we could start a marketing campaign for this specific group promoting “Household” insurance since they would want to have 
an insured home for their kids. And, as they are “Bad” Clients, we should give them incentive instead of penalties since they have lower income.

Based on the clusters obtained in the previous sections these are the marketing techniques recommended:
First of all, it is important to ensure that customers that produce most of the revenue for the company are kept and therefore, it can be suggested that for“Ideal” and “Stable” a loyalty program is introduced.
Moreover, it can be observed  that in clusters “ideal” and “potentially ideal” the distribution of spendings across different LOB is less equal and therefore, packages of product with a discount could be introduced: for example MOTOR and WORK COMPENSATION can be combined (i.e. one can break an arm in an accident and temporarily lose the ability to work ), MOTOR and HEALTH (one can break a leg in accident and need a specific treatment), etc.
Although “good enough” customers tend to spend very small fraction on LIFE and WORK COMPENSATION and it is not recommended to invest a significant resources into “Packages” technique since disposable income of these customer is relatively low.

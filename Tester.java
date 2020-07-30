import java.io.*;
import java.util.*;
//Program tests neural network
public class Tester
{
    /*************************************************************************/
    //constants for file conversions
    private static final int TRAIN_FILE = 1;
    private static final int VALIDATE_FILE = 2;
    private static final int TEST_FILE = 3;
    private static final int CLASSIFY_FILE = 4;
    //number of attributes
    private static final int ATTRIBUTES = 5;
    /*************************************************************************/
    //main method
    public static void main(String[] args) throws IOException
    {
        //preprocess files
        convert("trainingfile", "training", TRAIN_FILE);
        convert("testfile", "test", TEST_FILE);
        convert("validation", "validate", VALIDATE_FILE);
        //construct neural network
        NeuralNetwork network = new NeuralNetwork();
        
        network.loadTrainingData("training");               //load training data
    
        network.setParameters(5, 10000, 3412, 0.9);         //set parameters of network
    
        network.train();                                     //train network
        network.print();                                     //print the weights and theta values
        network.trainingError("training");                   //training error
        network.validate("validate");                        //validation error
        
        network.test("test", "outputfile");                    //test network
       
        convert("outputfile", "classifiedfile", CLASSIFY_FILE);      //postprocess files
    }
    /*************************************************************************/
    //Method converts one file to another file
    private static void convert(String inputFile, String outputFile, int fileType)throws IOException
    {
        //input and output files
        Scanner inFile = new Scanner(new File(inputFile));
        PrintWriter outFile = new PrintWriter(new FileWriter(outputFile));
        //convert original training file to training file
        if (fileType == TRAIN_FILE)
        {
            //read number of records, input file, output file
            int numberRecords = inFile.nextInt();
            int numberInputs = inFile.nextInt();
            int numberOutputs = inFile.nextInt();
            
            //write number of records, attributes, classes, neighbors, majority rule
            outFile.println(numberRecords + " " + numberInputs + " " + numberOutputs);
            
            //for each record
            for (int i = 0; i < numberRecords; i++)
            {
                //read attributes/classes and convert them to numbers
                for (int j = 0; j < ATTRIBUTES+1; j++)
                {
                    //read attribute/class label
                    String label = inFile.next();
                    //convert label to number
                    double value = convert(label, j);
                    //print attribute number
                    if (j < ATTRIBUTES)
                        outFile.print(value + " ");
                    //print class number
                    else
                        outFile.print((double)value + " ");
                }
                outFile.println();
            }
        }
        //convert original validation file to validation file
        else if (fileType == VALIDATE_FILE)
        {
            //read number of records
            int numberRecords = inFile.nextInt();
            //write number of records
            outFile.println(numberRecords);
            //for each record
            for (int i = 0; i < numberRecords; i++)
            {
                //read attributes/classes and convert them to numbers
                for (int j = 0; j < ATTRIBUTES+1; j++)
                {
                    //read attribute/class label
                    String label = inFile.next();
                    //convert label to number
                    double value = convert(label, j);
                    //print attribute number
                    if (j < ATTRIBUTES)
                        outFile.print(value + " ");
                    //print class number
                    else
                        outFile.print((double)value + " ");
                }
                outFile.println();
            }
        }
        //convert original test file to test file
        else if (fileType == TEST_FILE)
        {
            //read number of records
            int numberRecords = inFile.nextInt();
            //write number of records
            outFile.println(numberRecords);
            //for each record
            for (int i = 0; i < numberRecords; i++)
            {

                //read attributes and convert them to numbers
                for (int j = 0; j < ATTRIBUTES; j++)
                {
                    //read attribute label
                    String label = inFile.next();
                    //convert label to number
                    double value = convert(label, j);
                    //print number
                    outFile.print((double)value + " ");
                }
                outFile.println();
            }
        }
        //convert classified file to original classified file
        else if (fileType == CLASSIFY_FILE)
        {
            //read number of records
            int numberRecords = inFile.nextInt();
            //write number of records
            outFile.println(numberRecords);
            //for each record
            for (int i = 0; i < numberRecords; i++)
            {
                //read class number
                Double value = inFile.nextDouble();
                //convert number to label
                String label = convert(value);
                //print label
                outFile.println(label);
            }
        }
        inFile.close();
        outFile.close();
    }
    /*************************************************************************/
    //Method converts attribute/class label located at a column
    //to numerical value, column index starts at 0
    private static double convert(String label, int column)
    {
       
        double value;
        //column 0 - convert score to normalised
        if(column == 0)
        {
            value = Double.valueOf(label) - 500.0;
            value = value/400.0;
        }
        //column 1 - convert income to normalized income
        else if (column == 1)
        {
            value = Double.valueOf(label) - 30.0;
            value = value/60.0;
        }
        //column 2 - convert age to normalized age
        else if (column == 2)
        {
            value = Double.valueOf(label) - 30.0;
            value = value/50.0;
        }
        //column 3 - convert sex to number
        else if (column == 3)
        {
            if (label.equals("male"))
                value = 0.0;
            else
                value = 1.0;
        }
        //column 4 - convert marital status to number
        else if (column == 4)
        {
            if (label.equals("single"))
                value = 0.3;
            else if (label.equals("married"))
                value = 0.6;
            else
                value = 0.9;
        }
        //column 3 - convert class to number
        else 
        {
            if (label.equals("low"))
                value = 0.3;
            else if(label.equals("medium"))
                value = 0.6;
            else if(label.equals("high"))
                value = 0.9;
            else
                value = 0.0;
        }
        
        return value;
    }
    /*************************************************************************/
    //Method to find the distance between value and class
     private static double distance(double u, double v)
    {
        double distance=0;

        distance= (u-v) * (u-v);
        distance = Math.sqrt(distance);
        return distance;
    }
    /*************************************************************************/
    //method to find min value of distances
    private static double min(double first, double second, double third, double fourth) 
    {
        return Math.min(Math.min(Math.min(first, second), third), fourth);
    }
    /*************************************************************************/
    //Method converts class integer to class label
    private static String convert(double value)
    {
        String label="";
        double first, second, third,fourth, finalvalue;

        first = distance(0.3,value);
        second = distance(0.6,value);
        third = distance(0.9,value);
        fourth = distance(0.0,value);
        finalvalue= min (first,second,third,fourth);

        if (finalvalue == first )
            label = "low";
        else if (finalvalue == second )
            label = "medium";
        else if(finalvalue == third )
            label = "high";
        else
            label = "undetermined";
    
        return label;
    }
}
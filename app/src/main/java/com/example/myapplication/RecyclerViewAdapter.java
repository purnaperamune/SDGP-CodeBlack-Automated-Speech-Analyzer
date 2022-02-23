package com.example.myapplication;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.util.ArrayList;

public class RecyclerViewAdapter extends RecyclerView.Adapter<RecyclerViewAdapter.CustomViewHolder> {


    Context context;
    ArrayList<File> stringArrayList;
    PlayAudioInterface playAudioInterface;

    public RecyclerViewAdapter(Context context, ArrayList<File> stringArrayList, PlayAudioInterface playAudioInterface) {
        this.context = context;
        this.stringArrayList = stringArrayList;
        this.playAudioInterface= playAudioInterface;
    }

    @NonNull
    @Override
    public CustomViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view= LayoutInflater.from(parent.getContext()).inflate(R.layout.layout_item, parent, false);
        return new CustomViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull CustomViewHolder holder, int position) {
        File file= stringArrayList.get(position);
        holder.textView.setText(file.getName());

        holder.itemView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playAudioInterface.playAudio(holder.getAdapterPosition());
            }
        });
    }

    @Override
    public int getItemCount() {
        return stringArrayList.size();
    }

    class CustomViewHolder extends RecyclerView.ViewHolder{
        TextView textView;
        public CustomViewHolder(@NonNull View itemView) {
            super(itemView);

            textView= itemView.findViewById(R.id.musicName);
        }
    }
}
